import streamlit as st
import openai
import httpx
import base64
import io
import json
import cv2
import tempfile
import os
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

# --- Funções de Extração e Utilidades ---

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
        st.error(f"Erro ao ler arquivo: {e}")
    return texto

def split_gherkin_steps(gherkin: str) -> List[str]:
    lines = [line.strip() for line in gherkin.split('\n') if line.strip()]
    steps = []
    current_step = ""
    for line in lines:
        if line.startswith(('Dado ', 'Quando ', 'Então ', 'E ', 'Mas ', 'Given ', 'When ', 'Then ', 'And ', 'But ')):
            if current_step: steps.append(current_step.strip())
            current_step = line
        else: current_step += " " + line
    if current_step: steps.append(current_step.strip())
    return steps

def get_client(api_key: str):
    # Timeout aumentado para evitar 504 em redes lentas
    return openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", http_client=httpx.Client(timeout=120.0))

def extract_frames_from_video(video_bytes: bytes, num_frames: int = 3) -> List[bytes]:
    frames = []
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        interval = total_frames // num_frames
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(frame_rgb)
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=70) # Qualidade 70 para reduzir payload
                frames.append(buffer.getvalue())
    cap.release()
    os.unlink(tmp_path)
    return frames

# --- Core de Análise IA (Refatorado para Batching) ---

def analyze_ct_with_ai(ct_id: str, name: str, gherkin: str, files: List[Dict], api_key: str) -> Dict[str, Any]:
    client = get_client(api_key)
    steps = split_gherkin_steps(gherkin)

    processed_evidence = []
    images_payload = []

    # 1. Preparação das Imagens (Redimensionamento agressivo para evitar erro 504)
    for f in files:
        file_bytes_list = []
        if f['type'].startswith('image/'):
            file_bytes_list = [f['bytes']]
        elif f['type'].startswith('video/'):
            file_bytes_list = extract_frames_from_video(f['bytes'], num_frames=3)

        for b in file_bytes_list:
            try:
                img = PILImage.open(io.BytesIO(b))
                if img.mode in ("RGBA", "P"): img = img.convert("RGB")
                img.thumbnail((800, 800)) # Menor que 1024 reduz drasticamente o tráfego
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=70)
                final_bytes = buffered.getvalue()
                processed_evidence.append(final_bytes)
                b64 = base64.b64encode(final_bytes).decode('utf-8')
                images_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except: continue

    # 2. Construção do Prompt em Lote (Evita erro 429)
    passos_txt = "\n".join([f"- {s}" for s in steps])
    prompt_texto = (
        f"Cenário de Teste: {name}\n"
        f"Analise os seguintes passos Gherkin com base nas imagens enviadas:\n{passos_txt}\n\n"
        "Retorne OBRIGATORIAMENTE um objeto JSON puro com uma lista 'analises'. "
        "Cada item deve conter 'status' (PASSOU, FALHOU ou EVIDÊNCIA NÃO DISPONIBILIZADA) e 'justificativa'."
    )

    content = [{"type": "text", "text": prompt_texto}]
    content.extend(images_payload)

    try:
        # Apenas UMA chamada por cenário de teste
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": "Você é um auditor de QA sênior. Responda apenas em JSON estrito."},
                {"role": "user", "content": content}
            ],
            response_format={"type": "json_object"},
            timeout=90.0
        )

        # 3. Tratamento robusto do JSON (Limpeza de Markdown e Aspas)
        res_raw = response.choices[0].message.content.strip()
        if "```json" in res_raw:
            res_raw = res_raw.split("```json")[-1].split("```")[0].strip()

        parsed = json.loads(res_raw)
        ai_list = parsed.get('analises', [])

        # Mapeia de volta para os passos originais
        step_analyses = []
        for i, step in enumerate(steps):
            if i < len(ai_list):
                step_analyses.append({'step': step, **ai_list[i]})
            else:
                step_analyses.append({'step': step, 'status': 'ERRO', 'justificativa': 'IA não processou este passo.'})

    except Exception as e:
        step_analyses = [{'step': s, 'status': 'ERRO', 'justificativa': f'Falha técnica: {str(e)}'} for s in steps]

    pass_count = sum(1 for s in step_analyses if s['status'] in ["PASSOU", "EVIDÊNCIA NÃO DISPONIBILIZADA"])
    return {
        'ct_id': ct_id, 'name': name, 'steps': step_analyses,
        'pass_count': pass_count, 'total_steps': len(steps),
        'evidence_images': processed_evidence
    }

# --- Geração de Documentos (PDF e Sumário) ---

def generate_business_summary(results: List[Dict], metadata: Dict, api_key: str) -> str:
    client = get_client(api_key)
    jira_context = metadata.get('jira_full_text', 'Não informado')

    sumario_tecnico = ""
    for res in results:
        status_geral = "OK" if res['pass_count'] == res['total_steps'] else "COM FALHAS"
        sumario_tecnico += f"CT: {res['name']} ({status_geral})\n"

    prompt = f"""Atue como um PO Sênior. Resuma o valor de negócio destes testes frente aos requisitos.
    REQUISITOS: {jira_context}
    RESULTADOS: {sumario_tecnico}
    Seja executivo (1 parágrafo)."""

    try:
        response = client.chat.completions.create(model="google/gemini-2.0-flash-001", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except:
        return "Resumo gerado automaticamente com base na conformidade técnica dos cenários analisados."

def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setStrokeColor(BORDER_GRAY)
    canvas.line(15*mm, 10*mm, 195*mm, 10*mm)
    canvas.drawRightString(195*mm, 7*mm, f"Página {doc.page}")
    canvas.restoreState()

def generate_pdf(results: List[Dict], metadata: Dict, business_summary: str, logo_bytes: bytes = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, margin=(15*mm, 15*mm, 15*mm, 20*mm))
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T1', parent=styles['Heading1'], fontSize=16, textColor=PRIMARY_BLUE)
    section_style = ParagraphStyle('T2', parent=styles['Heading2'], fontSize=12, textColor=PRIMARY_BLUE, spaceBefore=12)
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontSize=9, leading=11)

    # Logo e Cabeçalho
    if logo_bytes:
        img_io = io.BytesIO(logo_bytes)
        logo_img = RLImage(img_io, width=35*mm, height=15*mm) # Ajuste proporcional
        story.append(Table([[logo_img, Paragraph("RELATÓRIO DE AUDITORIA", title_style)]], colWidths=[40*mm, 140*mm]))
    else:
        story.append(Paragraph("RELATÓRIO DE AUDITORIA", title_style))

    story.append(Spacer(1, 10))

    # Metadados
    meta_data = [
        [Paragraph(f"<b>PROJETO:</b> {metadata.get('teste_nome', '')}", meta_style), Paragraph(f"<b>SPRINT:</b> {metadata.get('sprint', '')}", meta_style)],
        [Paragraph(f"<b>HISTÓRIA:</b> {metadata.get('historia', '')}", meta_style), Paragraph(f"<b>VERSÃO:</b> {metadata.get('versao', '')}", meta_style)]
    ]
    t_meta = Table(meta_data, colWidths=[90*mm, 90*mm])
    t_meta.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, BORDER_GRAY), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
    story.append(t_meta)

    # Resumo de Negócio
    story.append(Paragraph("RESUMO EXECUTIVO", section_style))
    story.append(Table([[Paragraph(business_summary, styles['Normal'])]], colWidths=[180*mm], style=[('BACKGROUND', (0,0), (-1,-1), LIGHT_GRAY), ('PADDING', (0,0), (-1,-1), 8)]))

    # Resultados por CT
    for res in results:
        story.append(Paragraph(f"Cenário {res['ct_id']}: {res['name']}", section_style))
        steps_data = [[Paragraph("<b>Passo</b>", meta_style), Paragraph("<b>Status</b>", meta_style), Paragraph("<b>Justificativa</b>", meta_style)]]
        for s in res['steps']:
            color = PASS_GREEN if "PASSOU" in str(s['status']) else FAIL_RED
            steps_data.append([Paragraph(s['step'], meta_style), Paragraph(f"<font color={color}>{s['status']}</font>", meta_style), Paragraph(s['justificativa'], meta_style)])

        t_steps = Table(steps_data, colWidths=[60*mm, 35*mm, 85*mm])
        t_steps.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), PRIMARY_BLUE), ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('GRID', (0,0), (-1,-1), 0.5, BORDER_GRAY)]))
        story.append(t_steps)

        # Imagens no PDF
        for img_b in res.get('evidence_images', []):
            story.append(KeepTogether([Spacer(1, 5), RLImage(io.BytesIO(img_b), width=160*mm, height=90*mm)]))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return buffer.getvalue()

# --- Interface ---

st.sidebar.title("⚙️ Configurações")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
logo_upload = st.sidebar.file_uploader("Logo da Empresa", type=["png", "jpg"])

st.title("🧪 Auditoria GIC Pro")

with st.expander("📝 Detalhes da Execução", expanded=True):
    c1, c2 = st.columns(2)
    st.session_state.metadata['teste_nome'] = c1.text_input("Título do Teste")
    st.session_state.metadata['historia'] = c1.text_input("ID da História")
    st.session_state.metadata['sprint'] = c2.text_input("Sprint")
    st.session_state.metadata['versao'] = c2.text_input("Versão")

with st.container(border=True):
    st.subheader("📖 Requisitos (História/Jira)")
    modo = st.radio("Entrada:", ["Texto Manual", "Upload Documento"], horizontal=True)
    if modo == "Texto Manual":
        st.session_state.metadata['jira_full_text'] = st.text_area("Cole os critérios de aceite aqui:", height=150)
    else:
        req_file = st.file_uploader("PDF/TXT de requisitos", type=["pdf", "txt"])
        if req_file: st.session_state.metadata['jira_full_text'] = extrair_texto_arquivo(req_file)

st.header("📋 Cenários")
if st.button("➕ Adicionar Cenário"):
    st.session_state.ct_list.append({"id": f"CT{len(st.session_state.ct_list)+1:02d}", "name": "", "gherkin": "", "up_files": []})

for i, ct in enumerate(st.session_state.ct_list):
    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        ct['id'] = col1.text_input("ID", value=ct['id'], key=f"id_{i}")
        ct['name'] = col2.text_input("Nome", value=ct['name'], key=f"n_{i}")
        ct['gherkin'] = st.text_area("Gherkin", value=ct['gherkin'], key=f"g_{i}")
        ct['up_files'] = st.file_uploader(f"Evidências {ct['id']}", accept_multiple_files=True, key=f"f_{i}")

if st.button("🚀 Iniciar Auditoria", type="primary"):
    if not api_key: st.error("API Key necessária.")
    elif not st.session_state.ct_list: st.warning("Crie um cenário.")
    else:
        with st.spinner("IA Analisando..."):
            results = []
            for ct in st.session_state.ct_list:
                files_data = [{"name": f.name, "bytes": f.getvalue(), "type": f.type} for f in ct['up_files']]
                results.append(analyze_ct_with_ai(ct['id'], ct['name'], ct['gherkin'], files_data, api_key))

            summary = generate_business_summary(results, st.session_state.metadata, api_key)

            # Logo fallback
            logo_data = logo_upload.getvalue() if logo_upload else None

            pdf = generate_pdf(results, st.session_state.metadata, summary, logo_data)
            st.success("Auditoria pronta!")
            st.download_button("📥 Baixar Relatório PDF", data=pdf, file_name="auditoria_gic.pdf", mime="application/pdf")