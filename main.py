import streamlit as st
import openai
import httpx
import base64
import io
import json
import cv2
import tempfile
import os
import time
import PyPDF2
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from PIL import Image as PILImage
from typing import List, Dict, Any

st.set_page_config(page_title="Auditoria de Qualidade GIC", layout="wide")

# Configurações de Estilo e Cores
PRIMARY_BLUE = colors.HexColor("#1F4E79")
LIGHT_GRAY = colors.HexColor("#F2F2F2")
BORDER_GRAY = colors.HexColor("#D9D9D9")
PASS_GREEN = colors.HexColor("#2E7D32")
FAIL_RED = colors.HexColor("#C62828")
WARN_ORANGE = colors.HexColor("#EF6C00")

if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'ct_list' not in st.session_state:
    st.session_state.ct_list = []

# --- FUNÇÕES DE UTILITÁRIO ---

def extrair_texto_arquivo(uploaded_file):
    texto = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                texto += page.extract_text() or ""
        elif uploaded_file.type == "text/plain":
            texto = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Erro ao ler ficheiro: {e}")
    return texto

def get_client(api_key: str):
    return openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", http_client=httpx.Client())

def extract_frames_from_video(video_bytes: bytes, num_frames: int = 8) -> List[bytes]:
    frames = []
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        interval = max(1, total_frames // num_frames)
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(frame_rgb)
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=85)
                frames.append(buffer.getvalue())
    cap.release()
    if os.path.exists(tmp_path): os.unlink(tmp_path)
    return frames

# --- FUNÇÕES DE INTELIGÊNCIA ARTIFICIAL ---

def analyze_all_cts_at_once(ct_list: List[Dict], api_key: str) -> List[Dict]:
    client = get_client(api_key)
    all_results = []

    for ct in ct_list:
        scenarios_context = f"--- ID: {ct['id']} | NOME: {ct['name']} ---\nPASSOS:\n{ct['gherkin']}\n"
        images_payload = []

        for f in ct.get('up_files', []):
            try:
                # MODO EXTREMO DE POUPANÇA: Apenas 3 frames por vídeo
                file_bytes_list = extract_frames_from_video(f.getvalue(), num_frames=3) if f.type.startswith('video/') else [f.getvalue()]

                for b in file_bytes_list:
                    img = PILImage.open(io.BytesIO(b))
                    if img.mode in ("RGBA", "P"): img = img.convert("RGB")

                    # Redução drástica de resolução e qualidade para evitar o 429
                    img.thumbnail((768, 768))
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=60)
                    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    images_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except Exception as e:
                print(f"Erro ao processar imagem para IA: {e}")
                continue

        prompt = f"""Atue como um QA. Analise o Cenário abaixo contra as evidências (imagens em anexo).

        CENÁRIO:
        {scenarios_context}

        REQUISITOS:
        1. Compare as imagens em anexo para validar os passos.
        2. Retorne APENAS o JSON:
        {{
          "ct_id": "{ct['id']}",
          "steps_analysis": [
            {{"step": "Passo", "status": "PASSOU/FALHOU/EVIDÊNCIA NÃO DISPONIBILIZADA", "justificativa": "Análise visual detalhada."}}
          ]
        }}"""

        content = [{"type": "text", "text": prompt}] + images_payload

        max_retries = 3
        success = False

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="google/gemini-2.0-flash-001",
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    timeout=120.0
                )
                data = json.loads(response.choices[0].message.content)
                result = data.get("results", data) if isinstance(data, dict) else data

                if isinstance(result, list) and len(result) > 0:
                    all_results.append(result[0])
                else:
                    all_results.append(result)

                success = True
                break
            except Exception as e:
                error_str = str(e)
                if any(code in error_str for code in ["429", "504", "502"]) and attempt < max_retries - 1:
                    # Esperas muito mais longas: 30s, 60s, 90s
                    wait_time = (attempt + 1) * 30
                    st.warning(f"O Provedor da IA bloqueou por excesso de carga. Tentativa {attempt+1}/{max_retries} em {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Erro ao analisar o {ct['id']}:\n{error_str}")
                    break

        if not success:
            all_results.append({
                "ct_id": ct['id'],
                "steps_analysis": [{"step": "Geral", "status": "ERRO", "justificativa": "O servidor da IA rejeitou a análise por limites técnicos de processamento."}]
            })

    return all_results

def generate_business_summary(results: List[Dict], metadata: Dict, api_key: str) -> str:
    client = get_client(api_key)
    jira_context = metadata.get('jira_full_text', 'Não informado')

    contexto_testes = f"Teste: {metadata.get('teste_nome')}\n"
    for res in results:
        contexto_testes += f"Cenário: {res['name']} - Status: {res['pass_count']}/{res['total_steps']} validados.\n"

    prompt = f"""Atue como um Product Owner sênior.
    Analise se os resultados dos testes abaixo condizem com os REQUISITOS DA HISTÓRIA fornecidos.

    REQUISITOS DA HISTÓRIA/ACEITE:
    {jira_context}

    RESULTADOS TÉCNICOS:
    {contexto_testes}

    Escreva um 'Resumo de Negócio' de 1 parágrafo focado no valor, conformidade com o requisito e segurança. Seja executivo."""

    # Adicionado Retry para estabilidade do Resumo
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                st.error(f"Falha ao gerar o Resumo de Negócio: {e}")
                return "Não foi possível gerar o resumo automático devido a uma falha de comunicação com a IA após múltiplas tentativas."

# --- FUNÇÕES DE PDF ---

def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setStrokeColor(BORDER_GRAY)
    canvas.line(15*mm, 10*mm, 195*mm, 10*mm)
    canvas.drawRightString(195*mm, 7*mm, f"Página {doc.page}")
    canvas.drawString(15*mm, 7*mm, "Auditoria de Qualidade GIC - Confidencial")
    canvas.restoreState()

def generate_pdf(results: List[Dict], metadata: Dict, business_summary: str, logo_bytes: bytes = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=15*mm, bottomMargin=20*mm)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=16, textColor=PRIMARY_BLUE)
    section_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=12, textColor=PRIMARY_BLUE, spaceBefore=12)
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontSize=9, leading=11)

    if logo_bytes:
        img_io = io.BytesIO(logo_bytes)
        pil_img = PILImage.open(img_io)
        aspect = pil_img.size[1] / pil_img.size[0]
        logo_img = RLImage(img_io, width=40*mm, height=40*mm*aspect)
        header_table = Table([[logo_img, Paragraph("AUDITORIA DE QUALIDADE", title_style)]], colWidths=[45*mm, 135*mm])
        header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(header_table)
    else: story.append(Paragraph("AUDITORIA DE QUALIDADE", title_style))

    story.append(Paragraph(f"<font size=10 color=gray>{metadata.get('teste_nome', '').upper()}</font>", styles['Normal']))
    story.append(Spacer(1, 10))

    meta_data = [
        [Paragraph(f"<b>MÓDULO:</b> {metadata.get('modulo', '')}", meta_style), Paragraph(f"<b>SPRINT:</b> {metadata.get('sprint', '')}", meta_style)],
        [Paragraph(f"<b>STATUS:</b> {metadata.get('status', '')}", meta_style), Paragraph(f"<b>VERSÃO:</b> {metadata.get('versao', '')}", meta_style)],
        [Paragraph(f"<b>HISTÓRIA:</b> {metadata.get('historia', '')}", meta_style), Paragraph(f"<b>AUTOMATIZADO:</b> {metadata.get('automatizado', '')}", meta_style)],
        [Paragraph(f"<b>PRÉ REQUISITOS:</b> {metadata.get('pre_requisitos', '')}", meta_style), Paragraph(f"<b>Qtd. CT:</b> {len(results)}", meta_style)]
    ]
    meta_table = Table(meta_data, colWidths=[90*mm, 90*mm])
    meta_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, BORDER_GRAY), ('VALIGN', (0,0), (-1,-1), 'TOP'), ('BOTTOMPADDING', (0,0), (-1,-1), 5), ('TOPPADDING', (0,0), (-1,-1), 5)]))
    story.append(meta_table)

    story.append(Paragraph("RESUMO EXECUTIVO (NEGÓCIO)", section_style))
    summary_table = Table([[Paragraph(business_summary, styles['Normal'])]], colWidths=[180*mm])
    summary_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), LIGHT_GRAY), ('BOX', (0,0), (-1,-1), 0.5, BORDER_GRAY), ('PADDING', (0,0), (-1,-1), 10)]))
    story.append(summary_table)

    for res in results:
        story.append(Paragraph(f"Cenário {res.get('ct_id', '')}: {res['name']}", section_style))
        steps_data = [[Paragraph("<b>Passo Gherkin</b>", meta_style), Paragraph("<b>Status</b>", meta_style), Paragraph("<b>Justificativa IA</b>", meta_style)]]

        for s in res['steps']:
            color = PASS_GREEN if "PASSOU" in s['status'] else (WARN_ORANGE if "DISPONIBILIZADA" in s['status'] else FAIL_RED)
            steps_data.append([Paragraph(s['step'], meta_style), Paragraph(f"<b><font color={color}>{s['status']}</font></b>", meta_style), Paragraph(s['justificativa'], meta_style)])

        t = Table(steps_data, colWidths=[65*mm, 40*mm, 75*mm])
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), PRIMARY_BLUE), ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('GRID', (0,0), (-1,-1), 0.5, colors.white), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GRAY]), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(t)

        # Correção da renderização de Imagens no PDF
        if res.get('evidence_images'):
            story.append(Spacer(1, 15))
            story.append(Paragraph(f"<b>EVIDÊNCIAS VISUAIS - {res.get('ct_id', '')}:</b>", meta_style))
            story.append(Spacer(1, 5))

            for img_bytes in res['evidence_images']:
                try:
                    img_io = io.BytesIO(img_bytes)
                    pil_img = PILImage.open(img_io)

                    if pil_img.mode in ("RGBA", "P"):
                        pil_img = pil_img.convert("RGB")

                    # Força a criação de um buffer limpo em JPEG para o ReportLab ler sem erros
                    clean_buf = io.BytesIO()
                    pil_img.save(clean_buf, format="JPEG")
                    clean_buf.seek(0) # IMPORTANTÍSSIMO para o ReportLab não falhar

                    w, h = pil_img.size
                    max_width = 150 * mm
                    aspect = h / float(w)
                    calc_height = max_width * aspect

                    # Limita a altura para não quebrar a página
                    if calc_height > 200 * mm:
                        calc_height = 200 * mm
                        max_width = calc_height / aspect

                    story.append(KeepTogether([
                        RLImage(clean_buf, width=max_width, height=calc_height),
                        Spacer(1, 10)
                    ]))
                except Exception as e:
                    print(f"Erro ao desenhar imagem no PDF: {e}")
                    continue
        else:
            story.append(Spacer(1, 5))
            story.append(Paragraph("<i>Nenhuma evidência visual anexada.</i>", meta_style))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return buffer.getvalue()

# --- INTERFACE STREAMLIT ---

st.sidebar.title("🛠️ Configurações")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
logo_upload = st.sidebar.file_uploader("🖼️ Upload do Logo", type=["png", "jpg", "jpeg"])

st.title("🧪 Auditoria de Qualidade Pro")

with st.expander("📝 Cabeçalho do Documento", expanded=True):
    c1, c2 = st.columns(2)
    st.session_state.metadata['teste_nome'] = c1.text_input("TESTE (Título)")
    st.session_state.metadata['modulo'] = c1.text_input("MÓDULO")
    st.session_state.metadata['historia'] = c1.text_input("HISTÓRIA")
    st.session_state.metadata['pre_requisitos'] = c1.text_area("PRÉ REQUISITOS")
    st.session_state.metadata['status'] = c2.selectbox("STATUS", ["Aprovada", "Reprovada", "Em Análise"])
    st.session_state.metadata['sprint'] = c2.text_input("SPRINT")
    st.session_state.metadata['versao'] = c2.text_input("VERSÃO")
    st.session_state.metadata['automatizado'] = c2.radio("AUTOMATIZADO", ["Não", "Sim"], horizontal=True)

with st.container(border=True):
    st.subheader("📖 Detalhamento da História / Requisitos")
    modo_jira = st.radio("Selecione a forma de entrada:", ["Escrever Descrição (Jira)", "Anexar Documento de Aceite"], horizontal=True)

    if modo_jira == "Escrever Descrição (Jira)":
        jira_desc = st.text_area("Descrição completa da História e Critérios de Aceitação:", height=200)
        st.session_state.metadata['jira_full_text'] = jira_desc
    else:
        jira_file = st.file_uploader("Faça o upload do PDF ou TXT com os requisitos:", type=["pdf", "txt"])
        if jira_file:
            st.session_state.metadata['jira_full_text'] = extrair_texto_arquivo(jira_file)

st.header("📋 Cenários de Teste")
if st.button("➕ Adicionar CT"):
    st.session_state.ct_list.append({"id": f"CT{len(st.session_state.ct_list)+1:02d}", "name": "", "gherkin": "", "files": []})

for i, ct in enumerate(st.session_state.ct_list):
    with st.container(border=True):
        col_id, col_name = st.columns([1, 4])
        ct['id'] = col_id.text_input("ID", value=ct['id'], key=f"id_{i}")
        ct['name'] = col_name.text_input("Nome do Cenário", value=ct['name'], key=f"name_{i}")
        ct['gherkin'] = st.text_area("Gherkin", value=ct['gherkin'], key=f"gh_{i}")
        ct['up_files'] = st.file_uploader(f"Evidências {ct['id']}", accept_multiple_files=True, key=f"f_{i}", type=["png", "jpg", "jpeg", "mp4", "mov", "avi"])

if st.button("🚀 Gerar Auditoria Completa", type="primary"):
    if not api_key: st.error("Insira a API Key.")
    elif not st.session_state.ct_list: st.warning("Adicione pelo menos um cenário.")
    else:
        with st.spinner("A IA está a analisar os cenários e evidências..."):
            raw_results = analyze_all_cts_at_once(st.session_state.ct_list, api_key)

            final_results = []
            for ct in st.session_state.ct_list:
                analysis_data = next((item for item in raw_results if item.get('ct_id') == ct['id']), None)
                steps_formatted = analysis_data.get('steps_analysis', []) if analysis_data else []

                if not steps_formatted:
                    steps_formatted = [{'step': 'N/A', 'status': 'ERRO', 'justificativa': 'Não processado.'}]

                # Tratamento robótico de imagens para que o PDF nunca falhe
                processed_images = []
                for f in ct.get('up_files', []):
                    try:
                        if f.type.startswith('image/'):
                            processed_images.append(f.getvalue())
                        elif f.type.startswith('video/'):
                            processed_images.extend(extract_frames_from_video(f.getvalue()))
                    except: continue

                final_results.append({
                    'ct_id': ct['id'], 'name': ct['name'], 'steps': steps_formatted,
                    'pass_count': sum(1 for s in steps_formatted if "PASSOU" in s['status']),
                    'total_steps': len(steps_formatted), 'evidence_images': processed_images
                })

            summary = generate_business_summary(final_results, st.session_state.metadata, api_key)
            logo_data = logo_upload.getvalue() if logo_upload else None
            pdf = generate_pdf(final_results, st.session_state.metadata, summary, logo_data)

            st.success("Auditoria Concluída!")
            st.download_button("📥 Baixar PDF Premium", data=pdf, file_name="auditoria_qa_gic.pdf", mime="application/pdf")