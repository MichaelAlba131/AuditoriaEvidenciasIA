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
from concurrent.futures import ThreadPoolExecutor, as_completed
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from PIL import Image as PILImage
from typing import List, Dict, Any

st.set_page_config(page_title="Auditoria de Qualidade GIC", layout="wide")

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
        st.error(f"erro ao ler ficheiro: {e}")
    return texto

def get_client(api_key: str):
    return openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", http_client=httpx.Client())

@st.cache_data
def extract_frames_from_video(video_bytes: bytes, num_frames: int = 5) -> List[bytes]:
    frames = []
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
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
                    pil_img.save(buffer, format="JPEG", quality=65)
                    frames.append(buffer.getvalue())
        cap.release()
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
    return frames

def compress_image_for_api(image_bytes: bytes, max_size: int = 768, quality: int = 65) -> str:
    try:
        img = PILImage.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except:
        return None

def analyze_single_ct(ct: Dict, api_key: str, progress_placeholder=None) -> Dict:
    client = get_client(api_key)
    scenarios_context = f"--- ID: {ct['id']} | NOME: {ct['name']} ---\nPASSOS:\n{ct['gherkin']}\n"
    images_payload = []
    processed_images = []

    for f in ct.get('up_files', []):
        try:
            file_bytes_list = extract_frames_from_video(f.getvalue(), num_frames=5) if f.type.startswith('video/') else [f.getvalue()]

            for b in file_bytes_list:
                b64 = compress_image_for_api(b, max_size=768, quality=65)
                if b64:
                    images_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    processed_images.append(b)
        except Exception:
            continue

    prompt = f"""Atue como um Especialista em Auditoria de QA.
        Analise o Cenário de Teste abaixo e valide cada passo usando EXCLUSIVAMENTE as imagens enviadas em anexo.

        CENÁRIO:
        {scenarios_context}

        INSTRUÇÕES DE ANÁLISE:
        1. As imagens em anexo SÃO as evidências. Observe detalhes como títulos de janelas, estados de botões e dados em tabelas.
        2. Para validar se um botão 'funcionou', procure por mudanças sutis na interface entre uma imagem e outra.
        3. Seja específico na justificativa: mencione o que viu na imagem que prova o resultado.

        SAÍDA OBRIGATÓRIA (JSON):
        {{
          "ct_id": "{ct['id']}",
          "steps_analysis": [
            {{"step": "Texto do passo", "status": "PASSOU/FALHOU/EVIDÊNCIA NÃO DISPONIBILIZADA", "justificativa": "Descrição técnica."}}
          ]
        }}"""

    content = [{"type": "text", "text": prompt}] + images_payload
    max_retries = 3
    result = None

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
                result = result[0]

            result['_processed_images'] = processed_images
            break
        except Exception as e:
            error_str = str(e)
            if any(code in error_str for code in ["429", "504", "502"]) and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
            else:
                result = {
                    "ct_id": ct['id'],
                    "steps_analysis": [{"step": "falha técnica", "status": "ERRO", "justificativa": f"erro na api: {error_str[:100]}"}],
                    "_processed_images": processed_images
                }
                break

    if not result:
        result = {
            "ct_id": ct['id'],
            "steps_analysis": [{"step": "falha técnica", "status": "ERRO", "justificativa": "timeout no cenário."}],
            "_processed_images": processed_images
        }

    return result

def analyze_all_cts_at_once(ct_list: List[Dict], api_key: str) -> List[Dict]:
    all_results = []
    max_workers = min(3, len(ct_list))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_ct, ct, api_key): ct['id'] for ct in ct_list}

        for future in as_completed(futures):
            try:
                all_results.append(future.result(timeout=180))
            except Exception:
                ct_id = futures[future]
                all_results.append({
                    "ct_id": ct_id,
                    "steps_analysis": [{"step": "falha técnica", "status": "ERRO", "justificativa": "erro de processamento paralelo."}],
                    "_processed_images": []
                })
    return all_results

def generate_business_summary(results: List[Dict], metadata: Dict, api_key: str) -> str:
    client = get_client(api_key)
    jira_context = metadata.get('jira_full_text', 'não informado')

    contexto_testes = f"Teste: {metadata.get('teste_nome')}\n"
    for res in results:
        contexto_testes += f"Cenário: {res.get('name', 'N/A')} - Sucesso: {res.get('pass_count', 0)}/{res.get('total_steps', 0)}\n"

    prompt = f"""Atue como Product Owner. Com base nos requisitos e nos resultados técnicos, escreva um Resumo de Negócio executivo (1 parágrafo).
    REQUISITOS: {jira_context}
    RESULTADOS: {contexto_testes}"""

    # adicionado o sistema de retry aqui também
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0
            )
            return response.choices[0].message.content
        except Exception:
            if attempt < 2:
                time.sleep((attempt + 1) * 3)
            else:
                return "não foi possível gerar o resumo executivo automaticamente devido a instabilidades na ia."

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
        story.append(Paragraph(f"Cenário {res.get('ct_id', '')}: {res.get('name', 'N/A')}", section_style))
        steps_data = [[Paragraph("<b>Passo Gherkin</b>", meta_style), Paragraph("<b>Status</b>", meta_style), Paragraph("<b>Justificativa IA</b>", meta_style)]]

        for s in res.get('steps', []):
            color = PASS_GREEN if "PASSOU" in s['status'] else (WARN_ORANGE if "DISPONIBILIZADA" in s['status'] else FAIL_RED)
            steps_data.append([Paragraph(s['step'], meta_style), Paragraph(f"<b><font color={color}>{s['status']}</font></b>", meta_style), Paragraph(s['justificativa'], meta_style)])

        t = Table(steps_data, colWidths=[65*mm, 40*mm, 75*mm])
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), PRIMARY_BLUE), ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('GRID', (0,0), (-1,-1), 0.5, colors.white), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GRAY]), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(t)

        if res.get('evidence_images'):
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>EVIDÊNCIAS VISUAIS:</b>", meta_style))
            for img_bytes in res['evidence_images']:
                try:
                    img_io = io.BytesIO(img_bytes)
                    pil_img = PILImage.open(img_io).convert("RGB")
                    clean_buf = io.BytesIO()
                    pil_img.save(clean_buf, format="JPEG")
                    clean_buf.seek(0)
                    w, h = pil_img.size
                    max_width = 165 * mm
                    aspect = h / float(w)
                    story.append(KeepTogether([Spacer(1, 5), RLImage(clean_buf, width=max_width, height=max_width * aspect), Spacer(1, 5)]))
                except: continue

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return buffer.getvalue()

st.sidebar.title("🛠️ Configurações")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
logo_upload = st.sidebar.file_uploader("🖼️ Logo da Empresa", type=["png", "jpg", "jpeg"])

st.title("🧪 Auditoria de Qualidade Pro")

with st.expander("📝 Cabeçalho do Documento", expanded=True):
    c1, c2 = st.columns(2)
    st.session_state.metadata['teste_nome'] = c1.text_input("Título do Teste")
    st.session_state.metadata['modulo'] = c1.text_input("Módulo")
    st.session_state.metadata['historia'] = c1.text_input("História/Ticket")
    st.session_state.metadata['pre_requisitos'] = c1.text_area("Pré-requisitos")

    # status manual inicial
    status_manual = c2.selectbox("Status Final", ["Aprovada", "Reprovada", "Em Análise"])
    st.session_state.metadata['status'] = status_manual

    st.session_state.metadata['sprint'] = c2.text_input("Sprint")
    st.session_state.metadata['versao'] = c2.text_input("Versão/Build")
    st.session_state.metadata['automatizado'] = c2.radio("Automatizado?", ["Não", "Sim"], horizontal=True)

with st.container(border=True):
    st.subheader("📖 Requisitos da História")
    modo_jira = st.radio("Entrada:", ["Texto Direto", "Upload de Arquivo"], horizontal=True)
    if modo_jira == "Texto Direto":
        st.session_state.metadata['jira_full_text'] = st.text_area("Descrição/Critérios de Aceite:", height=150)
    else:
        jira_file = st.file_uploader("Arquivo de requisitos:", type=["pdf", "txt"])
        if jira_file: st.session_state.metadata['jira_full_text'] = extrair_texto_arquivo(jira_file)

st.header("📋 Cenários")
if st.button("➕ Adicionar Cenário"):
    st.session_state.ct_list.append({"id": f"CT{len(st.session_state.ct_list)+1:02d}", "name": "", "gherkin": "", "up_files": []})

for i, ct in enumerate(st.session_state.ct_list):
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        ct['id'] = col1.text_input("ID", value=ct['id'], key=f"id_{i}")
        ct['name'] = col2.text_input("Cenário", value=ct['name'], key=f"name_{i}")
        ct['gherkin'] = st.text_area("Passos (Gherkin)", value=ct['gherkin'], key=f"gh_{i}")
        ct['up_files'] = st.file_uploader(f"Anexar Provas ({ct['id']})", accept_multiple_files=True, key=f"f_{i}")

if st.button("🚀 Gerar Auditoria", type="primary"):
    if not api_key: st.error("Falta a API Key.")
    elif not st.session_state.ct_list: st.warning("Adicione cenários.")
    else:
        with st.spinner("analisando evidências..."):
            raw_results = analyze_all_cts_at_once(st.session_state.ct_list, api_key)

            final_results = []
            results_map = {item.get('ct_id'): item for item in raw_results}

            total_passos_gerais = 0
            total_passos_sucesso = 0
            tem_erro = False

            for ct in st.session_state.ct_list:
                analysis = results_map.get(ct['id'], {})
                steps = analysis.get('steps_analysis', [])
                images = analysis.get('_processed_images', [])

                passos_sucesso_ct = sum(1 for s in steps if "PASSOU" in s.get('status', ''))

                # checa se a ia devolveu algum erro técnico ou falha
                if any(s.get('status') in ["ERRO", "FALHOU", "EVIDÊNCIA NÃO DISPONIBILIZADA"] for s in steps):
                    tem_erro = True

                total_passos_gerais += len(steps)
                total_passos_sucesso += passos_sucesso_ct

                final_results.append({
                    'ct_id': ct['id'], 'name': ct['name'], 'steps': steps,
                    'pass_count': passos_sucesso_ct,
                    'total_steps': len(steps), 'evidence_images': images
                })

            # sobrescreve o status manual com base na realidade técnica da ia
            if tem_erro:
                st.session_state.metadata['status'] = "Reprovada (Com Falhas/Erros)"
            elif total_passos_sucesso == total_passos_gerais and total_passos_gerais > 0:
                st.session_state.metadata['status'] = "Aprovada"

            sum_text = generate_business_summary(final_results, st.session_state.metadata, api_key)
            pdf = generate_pdf(final_results, st.session_state.metadata, sum_text, logo_upload.getvalue() if logo_upload else None)

            st.success("pronto!")
            st.download_button("📥 Baixar PDF de Auditoria", data=pdf, file_name="auditoria_gic.pdf", mime="application/pdf")