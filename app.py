from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st
from agents import Agent, Runner, function_tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except ImportError:  # pragma: no cover - optional dependency
    convert_from_bytes = None

APP_TITLE = "Patient Intelligence Console"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "patients.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "patients.db")

ACTIVE_HOSPITAL_NUMBER: str | None = None


def load_patients_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def init_db(db_path: str, seed_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                hospital_number TEXT UNIQUE NOT NULL,
                patient_name TEXT NOT NULL,
                blood_pressure TEXT NOT NULL,
                fever TEXT NOT NULL,
                heart_rate TEXT NOT NULL,
                temperature TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                entry TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS doctor_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                note_type TEXT NOT NULL,
                entry TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                ocr_confidence REAL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """
        )
        cursor.execute("PRAGMA table_info(patient_documents)")
        columns = {row[1] for row in cursor.fetchall()}
        if "ocr_confidence" not in columns:
            cursor.execute(
                "ALTER TABLE patient_documents ADD COLUMN ocr_confidence REAL"
            )
        cursor.execute("SELECT COUNT(*) FROM patients")
        patient_count = cursor.fetchone()[0]
        if patient_count == 0:
            patients = load_patients_from_json(seed_path)
            for patient in patients:
                cursor.execute(
                    """
                    INSERT INTO patients (
                        patient_id,
                        hospital_number,
                        patient_name,
                        blood_pressure,
                        fever,
                        heart_rate,
                        temperature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient["patient_id"],
                        patient["hospital_number"],
                        patient["patient_name"],
                        patient["blood_pressure"],
                        patient["fever"],
                        patient["heart_rate"],
                        patient["temperature"],
                    ),
                )
                for entry in patient.get("history", []):
                    cursor.execute(
                        """
                        INSERT INTO patient_history (patient_id, entry)
                        VALUES (?, ?)
                        """,
                        (patient["patient_id"], entry),
                    )
                for note in patient.get("doctor_notes", []):
                    cursor.execute(
                        """
                        INSERT INTO doctor_notes (patient_id, note_type, entry)
                        VALUES (?, ?, ?)
                        """,
                        (
                            patient["patient_id"],
                            note.get("type", "digital"),
                            note.get("entry", ""),
                        ),
                    )
                for doc in patient.get("documents", []):
                    cursor.execute(
                        """
                        INSERT INTO patient_documents (
                            patient_id,
                            doc_type,
                            title,
                            content,
                            ocr_confidence
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            patient["patient_id"],
                            doc.get("type", "digital"),
                            doc.get("title", "Untitled Document"),
                            doc.get("content", ""),
                            doc.get("ocr_confidence"),
                        ),
                    )
            conn.commit()
    finally:
        conn.close()


def fetch_patients(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_id,
                   hospital_number,
                   patient_name,
                   blood_pressure,
                   fever,
                   heart_rate,
                   temperature
            FROM patients
            ORDER BY patient_name
            """
        )
        rows = cursor.fetchall()
        patients = [dict(row) for row in rows]
        for patient in patients:
            patient["history"] = fetch_patient_history(
                db_path, patient["patient_id"]
            )
            patient["doctor_notes"] = fetch_doctor_notes(
                db_path, patient["patient_id"]
            )
            patient["documents"] = fetch_patient_documents(
                db_path, patient["patient_id"]
            )
        return patients
    finally:
        conn.close()


def fetch_patient_history(db_path: str, patient_id: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT entry
            FROM patient_history
            WHERE patient_id = ?
            ORDER BY id
            """,
            (patient_id,),
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def fetch_doctor_notes(
    db_path: str, patient_id: str
) -> List[Dict[str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT note_type, entry
            FROM doctor_notes
            WHERE patient_id = ?
            ORDER BY id
            """,
            (patient_id,),
        )
        rows = cursor.fetchall()
        return [{"type": row[0], "entry": row[1]} for row in rows]
    finally:
        conn.close()


def fetch_patient_documents(
    db_path: str, patient_id: str
) -> List[Dict[str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT doc_type, title, content, ocr_confidence
            FROM patient_documents
            WHERE patient_id = ?
            ORDER BY id
            """,
            (patient_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "type": row[0],
                "title": row[1],
                "content": row[2],
                "ocr_confidence": row[3],
            }
            for row in rows
        ]
    finally:
        conn.close()


def add_doctor_notes(
    db_path: str, hospital_number: str, notes: List[Dict[str, str]]
) -> Tuple[int, int]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_id
            FROM patients
            WHERE hospital_number = ?
            """,
            (hospital_number,),
        )
        row = cursor.fetchone()
        if not row:
            return (0, len(notes))
        patient_id = row[0]
        inserted = 0
        for note in notes:
            cursor.execute(
                """
                INSERT INTO doctor_notes (patient_id, note_type, entry)
                VALUES (?, ?, ?)
                """,
                (
                    patient_id,
                    note.get("type", "digital"),
                    note.get("entry", ""),
                ),
            )
            inserted += 1
        conn.commit()
        return (inserted, 0)
    finally:
        conn.close()


def add_documents(
    db_path: str, hospital_number: str, documents: List[Dict[str, str]]
) -> Tuple[int, int]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_id
            FROM patients
            WHERE hospital_number = ?
            """,
            (hospital_number,),
        )
        row = cursor.fetchone()
        if not row:
            return (0, len(documents))
        patient_id = row[0]
        inserted = 0
        for doc in documents:
            cursor.execute(
                """
                INSERT INTO patient_documents (
                    patient_id,
                    doc_type,
                    title,
                    content,
                    ocr_confidence
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    doc.get("type", "digital"),
                    doc.get("title", "Untitled Document"),
                    doc.get("content", ""),
                    doc.get("ocr_confidence"),
                ),
            )
            inserted += 1
        conn.commit()
        return (inserted, 0)
    finally:
        conn.close()


def create_patient(db_path: str, patient: Dict[str, Any]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO patients (
                patient_id,
                hospital_number,
                patient_name,
                blood_pressure,
                fever,
                heart_rate,
                temperature
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient["patient_id"],
                patient["hospital_number"],
                patient["patient_name"],
                patient["blood_pressure"],
                patient["fever"],
                patient["heart_rate"],
                patient["temperature"],
            ),
        )
        for entry in patient.get("history", []):
            cursor.execute(
                """
                INSERT INTO patient_history (patient_id, entry)
                VALUES (?, ?)
                """,
                (patient["patient_id"], entry),
            )
        for note in patient.get("doctor_notes", []):
            cursor.execute(
                """
                INSERT INTO doctor_notes (patient_id, note_type, entry)
                VALUES (?, ?, ?)
                """,
                (
                    patient["patient_id"],
                    note.get("type", "digital"),
                    note.get("entry", ""),
                ),
            )
        for doc in patient.get("documents", []):
            cursor.execute(
                """
                INSERT INTO patient_documents (
                    patient_id,
                    doc_type,
                    title,
                    content,
                    ocr_confidence
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    patient["patient_id"],
                    doc.get("type", "digital"),
                    doc.get("title", "Untitled Document"),
                    doc.get("content", ""),
                    doc.get("ocr_confidence"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def update_patient(db_path: str, patient: Dict[str, Any]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE patients
            SET patient_name = ?,
                blood_pressure = ?,
                fever = ?,
                heart_rate = ?,
                temperature = ?
            WHERE hospital_number = ?
            """,
            (
                patient["patient_name"],
                patient["blood_pressure"],
                patient["fever"],
                patient["heart_rate"],
                patient["temperature"],
                patient["hospital_number"],
            ),
        )
        cursor.execute(
            """
            DELETE FROM patient_history
            WHERE patient_id = ?
            """,
            (patient["patient_id"],),
        )
        cursor.execute(
            """
            DELETE FROM doctor_notes
            WHERE patient_id = ?
            """,
            (patient["patient_id"],),
        )
        cursor.execute(
            """
            DELETE FROM patient_documents
            WHERE patient_id = ?
            """,
            (patient["patient_id"],),
        )
        for entry in patient.get("history", []):
            cursor.execute(
                """
                INSERT INTO patient_history (patient_id, entry)
                VALUES (?, ?)
                """,
                (patient["patient_id"], entry),
            )
        for note in patient.get("doctor_notes", []):
            cursor.execute(
                """
                INSERT INTO doctor_notes (patient_id, note_type, entry)
                VALUES (?, ?, ?)
                """,
                (
                    patient["patient_id"],
                    note.get("type", "digital"),
                    note.get("entry", ""),
                ),
            )
        for doc in patient.get("documents", []):
            cursor.execute(
                """
                INSERT INTO patient_documents (
                    patient_id,
                    doc_type,
                    title,
                    content,
                    ocr_confidence
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    patient["patient_id"],
                    doc.get("type", "digital"),
                    doc.get("title", "Untitled Document"),
                    doc.get("content", ""),
                    doc.get("ocr_confidence"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def parse_history(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_notes(text: str) -> List[Dict[str, str]]:
    notes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if "|" in line:
            note_type, entry = line.split("|", 1)
            notes.append(
                {"type": note_type.strip().lower(), "entry": entry.strip()}
            )
        else:
            notes.append({"type": "digital", "entry": line.strip()})
    return notes


def parse_documents(text: str) -> List[Dict[str, str]]:
    docs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) == 3:
            doc_type, title, content = parts
        elif len(parts) == 2:
            doc_type, title = parts
            content = ""
        else:
            doc_type = "digital"
            title = "Untitled Document"
            content = parts[0]
        docs.append(
            {
                "type": normalize_note_type(doc_type),
                "title": title or "Untitled Document",
                "content": content,
            }
        )
    return docs


def normalize_note_type(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned not in {"digital", "handwritten"}:
        return "digital"
    return cleaned


def parse_notes_upload(
    filename: str, raw_bytes: bytes
) -> List[Dict[str, str]]:
    if filename.lower().endswith(".json"):
        payload = json.loads(raw_bytes.decode("utf-8"))
        notes = []
        for item in payload:
            notes.append(
                {
                    "hospital_number": item.get("hospital_number", "").strip(),
                    "type": normalize_note_type(item.get("type", "digital")),
                    "entry": item.get("entry", "").strip(),
                }
            )
        return [note for note in notes if note["hospital_number"]]
    if filename.lower().endswith(".csv"):
        text = raw_bytes.decode("utf-8").splitlines()
        reader = csv.DictReader(text)
        notes = []
        for row in reader:
            notes.append(
                {
                    "hospital_number": row.get("hospital_number", "").strip(),
                    "type": normalize_note_type(row.get("type", "digital")),
                    "entry": row.get("entry", "").strip(),
                }
            )
        return [note for note in notes if note["hospital_number"]]
    return []


def parse_documents_upload(
    filename: str, raw_bytes: bytes
) -> List[Dict[str, str]]:
    if filename.lower().endswith(".json"):
        payload = json.loads(raw_bytes.decode("utf-8"))
        docs = []
        for item in payload:
            docs.append(
                {
                    "hospital_number": item.get("hospital_number", "").strip(),
                    "type": normalize_note_type(item.get("type", "digital")),
                    "title": item.get("title", "Untitled Document").strip(),
                    "content": item.get("content", "").strip(),
                }
            )
        return [doc for doc in docs if doc["hospital_number"]]
    if filename.lower().endswith(".csv"):
        text = raw_bytes.decode("utf-8").splitlines()
        reader = csv.DictReader(text)
        docs = []
        for row in reader:
            docs.append(
                {
                    "hospital_number": row.get("hospital_number", "").strip(),
                    "type": normalize_note_type(row.get("type", "digital")),
                    "title": row.get("title", "Untitled Document").strip(),
                    "content": row.get("content", "").strip(),
                }
            )
        return [doc for doc in docs if doc["hospital_number"]]
    return []


def ocr_image(image: Image.Image) -> Tuple[str, float | None]:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed.")
    text = pytesseract.image_to_string(image).strip()
    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT
    )
    confidences = []
    for raw in data.get("conf", []):
        try:
            value = float(raw)
        except (ValueError, TypeError):
            continue
        if value >= 0:
            confidences.append(value)
    confidence = None
    if confidences:
        confidence = sum(confidences) / len(confidences)
    return text, confidence


def ocr_from_upload(filename: str, raw_bytes: bytes) -> Tuple[str, float | None]:
    if filename.lower().endswith(".pdf"):
        if convert_from_bytes is None:
            raise RuntimeError("pdf2image is not installed.")
        pages = convert_from_bytes(raw_bytes)
        text_chunks = []
        confidence_values = []
        for page in pages:
            text, confidence = ocr_image(page)
            if text:
                text_chunks.append(text)
            if confidence is not None:
                confidence_values.append(confidence)
        combined_text = "\n".join(text_chunks).strip()
        combined_confidence = None
        if confidence_values:
            combined_confidence = sum(confidence_values) / len(
                confidence_values
            )
        return combined_text, combined_confidence
    image = Image.open(io.BytesIO(raw_bytes))
    return ocr_image(image)


def build_patient_documents(patient: Dict[str, Any]) -> List[Document]:
    profile_text = (
        "Patient Profile\n"
        f"Patient ID: {patient['patient_id']}\n"
        f"Hospital Number: {patient['hospital_number']}\n"
        f"Patient Name: {patient['patient_name']}\n"
        f"Blood Pressure: {patient['blood_pressure']}\n"
        f"Fever: {patient['fever']}\n"
        f"Heart Rate: {patient['heart_rate']}\n"
        f"Temperature: {patient['temperature']}\n"
    )
    docs = [
        Document(
            page_content=profile_text,
            metadata={
                "hospital_number": patient["hospital_number"],
                "patient_id": patient["patient_id"],
                "doc_type": "profile",
            },
        )
    ]
    for note in patient.get("doctor_notes", []):
        docs.append(
            Document(
                page_content=(
                    "Doctor Note "
                    f"({note.get('type', 'digital')}): "
                    f"{note.get('entry', '')}"
                ),
                metadata={
                    "hospital_number": patient["hospital_number"],
                    "patient_id": patient["patient_id"],
                    "doc_type": "doctor_note",
                },
            )
        )
    for doc in patient.get("documents", []):
        docs.append(
            Document(
                page_content=(
                    "Patient Document "
                    f"({doc.get('type', 'digital')}): "
                    f"{doc.get('title', '')}. {doc.get('content', '')}"
                ),
                metadata={
                    "hospital_number": patient["hospital_number"],
                    "patient_id": patient["patient_id"],
                    "doc_type": "document",
                },
            )
        )
    for entry in patient.get("history", []):
        docs.append(
            Document(
                page_content=(
                    f"Patient History ({patient['hospital_number']}): {entry}"
                ),
                metadata={
                    "hospital_number": patient["hospital_number"],
                    "patient_id": patient["patient_id"],
                    "doc_type": "history",
                },
            )
        )
    return docs


def build_vectorstores(
    patients: List[Dict[str, Any]]
) -> Dict[str, FAISS]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=40,
    )
    embeddings = OpenAIEmbeddings()
    stores: Dict[str, FAISS] = {}
    for patient in patients:
        docs = build_patient_documents(patient)
        chunks = splitter.split_documents(docs)
        stores[patient["hospital_number"]] = FAISS.from_documents(
            chunks, embeddings
        )
    return stores


def enforce_hospital_number(hospital_number: str) -> Dict[str, Any] | None:
    if ACTIVE_HOSPITAL_NUMBER is None:
        return {"error": "No active hospital number set."}
    if hospital_number != ACTIVE_HOSPITAL_NUMBER:
        return {
            "error": (
                "Hospital number mismatch. Access is restricted to the "
                f"selected profile {ACTIVE_HOSPITAL_NUMBER}."
            )
        }
    return None


@function_tool
def get_patient_profile(hospital_number: str) -> Dict[str, Any]:
    violation = enforce_hospital_number(hospital_number)
    if violation:
        return violation
    patient = PATIENT_MAP.get(hospital_number)
    if not patient:
        return {"error": "Patient not found."}
    return patient


@function_tool
def search_patient_history(hospital_number: str, query: str) -> Dict[str, Any]:
    violation = enforce_hospital_number(hospital_number)
    if violation:
        return violation
    if not query.strip():
        return {"error": "Query is empty."}
    store = VECTORSTORES.get(hospital_number)
    if not store:
        return {"error": "No vector store for this patient."}
    results = store.similarity_search(query, k=4)
    return {
        "matches": [doc.page_content for doc in results],
    }


def build_agent(hospital_number: str) -> Agent:
    instructions = (
        "You are a clinical data assistant. You must only answer questions "
        f"for hospital number {hospital_number}. Do not invent or reference "
        "any other hospital numbers. Use the tools to retrieve patient "
        "profile and history. If the tools return an error, explain the "
        "issue and stop. Never use external sources or internet searches."
    )
    return Agent(
        name="PatientRecordsAgent",
        instructions=instructions,
        tools=[get_patient_profile, search_patient_history],
    )


def ask_agent(question: str, hospital_number: str) -> str:
    global ACTIVE_HOSPITAL_NUMBER
    ACTIVE_HOSPITAL_NUMBER = hospital_number
    agent = build_agent(hospital_number)
    result = Runner.run_sync(
        agent, question
    )
    ACTIVE_HOSPITAL_NUMBER = None
    return result.final_output


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Patient Intelligence Console")
st.markdown(
    "Curated mock patient profiles from SQLite with "
    "retrieval-augmented answers grounded in the selected hospital "
    "number."
)

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "Set the `OPENAI_API_KEY` environment variable to enable embeddings "
        "and agent responses."
    )
    st.stop()

init_db(DB_PATH, DATA_PATH)
PATIENTS = fetch_patients(DB_PATH)
PATIENT_MAP = {patient["hospital_number"]: patient for patient in PATIENTS}
VECTORSTORES = build_vectorstores(PATIENTS)

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

left_col, right_col = st.columns([1.1, 1.4], gap="large")

with left_col:
    st.subheader("Patient Profiles")
    selection = st.selectbox(
        "Select a patient profile",
        options=PATIENTS,
        format_func=lambda p: f"{p['patient_name']} · {p['hospital_number']}",
    )
    hospital_number = selection["hospital_number"]

    st.markdown("### Selected Profile")
    st.write(f"**Patient Name:** {selection['patient_name']}")
    st.write(f"**Patient ID:** {selection['patient_id']}")
    st.write(f"**Hospital Number:** {selection['hospital_number']}")

    vitals_row = st.columns(2)
    with vitals_row[0]:
        st.metric("Blood Pressure", selection["blood_pressure"])
        st.metric("Heart Rate", selection["heart_rate"])
    with vitals_row[1]:
        st.metric("Temperature", selection["temperature"])
        st.metric("Fever", selection["fever"])

    with st.expander("Patient History", expanded=True):
        for entry in selection.get("history", []):
            st.markdown(f"- {entry}")

    with st.expander("Doctor Notes (Digital + Handwritten)", expanded=True):
        for note in selection.get("doctor_notes", []):
            note_type = note.get("type", "digital").title()
            st.markdown(f"- **{note_type}:** {note.get('entry', '')}")

    with st.expander("Patient Documents (Digital + Handwritten)", expanded=False):
        for doc in selection.get("documents", []):
            doc_type = doc.get("type", "digital").title()
            title = doc.get("title", "Untitled Document")
            content = doc.get("content", "")
            confidence = doc.get("ocr_confidence")
            confidence_text = ""
            if confidence is not None:
                confidence_text = f" (OCR confidence {confidence:.1f}%)"
            st.markdown(
                f"- **{doc_type} · {title}{confidence_text}:** {content}"
            )

    st.divider()
    st.subheader("Admin")
    add_tab, edit_tab = st.tabs(["Add Patient", "Edit Patient"])

    with add_tab:
        with st.form("add_patient_form"):
            st.caption("Create a new patient profile in SQLite.")
            add_patient_id = st.text_input("Patient ID", key="add_patient_id")
            add_hospital_number = st.text_input(
                "Hospital Number", key="add_hospital_number"
            )
            add_name = st.text_input("Patient Name", key="add_name")
            add_bp = st.text_input("Blood Pressure", key="add_bp")
            add_fever = st.text_input("Fever", key="add_fever")
            add_hr = st.text_input("Heart Rate", key="add_hr")
            add_temp = st.text_input("Temperature", key="add_temp")
            add_history = st.text_area(
                "History (one entry per line)", key="add_history"
            )
            add_notes = st.text_area(
                "Doctor Notes (type|note per line)",
                key="add_notes",
                help=(
                    "Format: digital|note or handwritten|note. "
                    "If no type provided, defaults to digital."
                ),
            )
            add_documents = st.text_area(
                "Patient Documents (type|title|content per line)",
                key="add_documents",
                help=(
                    "Format: digital|title|content or handwritten|title|content."
                ),
            )
            add_submit = st.form_submit_button("Add Patient")

        if add_submit:
            if not all(
                [
                    add_patient_id.strip(),
                    add_hospital_number.strip(),
                    add_name.strip(),
                ]
            ):
                st.error("Patient ID, hospital number, and name are required.")
            else:
                payload = {
                    "patient_id": add_patient_id.strip(),
                    "hospital_number": add_hospital_number.strip(),
                    "patient_name": add_name.strip(),
                    "blood_pressure": add_bp.strip(),
                    "fever": add_fever.strip(),
                    "heart_rate": add_hr.strip(),
                    "temperature": add_temp.strip(),
                    "history": parse_history(add_history),
                    "doctor_notes": parse_notes(add_notes),
                    "documents": parse_documents(add_documents),
                }
                try:
                    create_patient(DB_PATH, payload)
                    st.success("Patient added.")
                    st.rerun()
                except sqlite3.IntegrityError as exc:
                    st.error(f"Unable to add patient: {exc}")

    with edit_tab:
        edit_choice = st.selectbox(
            "Select patient to edit",
            options=PATIENTS,
            format_func=lambda p: f"{p['patient_name']} · {p['hospital_number']}",
            key="edit_patient_select",
        )
        with st.form("edit_patient_form"):
            st.caption("Update profile details and replace history.")
            edit_name = st.text_input(
                "Patient Name", value=edit_choice["patient_name"]
            )
            edit_bp = st.text_input(
                "Blood Pressure", value=edit_choice["blood_pressure"]
            )
            edit_fever = st.text_input("Fever", value=edit_choice["fever"])
            edit_hr = st.text_input(
                "Heart Rate", value=edit_choice["heart_rate"]
            )
            edit_temp = st.text_input(
                "Temperature", value=edit_choice["temperature"]
            )
            edit_history = st.text_area(
                "History (one entry per line)",
                value="\n".join(edit_choice.get("history", [])),
            )
            edit_notes = st.text_area(
                "Doctor Notes (type|note per line)",
                value="\n".join(
                    [
                        f"{note.get('type', 'digital')}|{note.get('entry', '')}"
                        for note in edit_choice.get("doctor_notes", [])
                    ]
                ),
            )
            edit_documents = st.text_area(
                "Patient Documents (type|title|content per line)",
                value="\n".join(
                    [
                        (
                            f"{doc.get('type', 'digital')}|"
                            f"{doc.get('title', '')}|"
                            f"{doc.get('content', '')}"
                        )
                        for doc in edit_choice.get("documents", [])
                    ]
                ),
            )
            edit_submit = st.form_submit_button("Update Patient")

        if edit_submit:
            payload = {
                "patient_id": edit_choice["patient_id"],
                "hospital_number": edit_choice["hospital_number"],
                "patient_name": edit_name.strip(),
                "blood_pressure": edit_bp.strip(),
                "fever": edit_fever.strip(),
                "heart_rate": edit_hr.strip(),
                "temperature": edit_temp.strip(),
                "history": parse_history(edit_history),
                "doctor_notes": parse_notes(edit_notes),
                "documents": parse_documents(edit_documents),
            }
            update_patient(DB_PATH, payload)
            st.success("Patient updated.")
            st.rerun()

    with st.expander("Import Doctor Notes", expanded=False):
        st.caption(
            "Upload CSV or JSON with columns: hospital_number, type, entry."
        )
        uploaded = st.file_uploader(
            "Upload notes",
            type=["csv", "json"],
            help="Rows without a matching hospital number are skipped.",
            key="notes_upload",
        )
        import_submit = st.button("Import Notes")
        if import_submit:
            if not uploaded:
                st.error("Please upload a CSV or JSON file.")
            else:
                notes = parse_notes_upload(
                    uploaded.name, uploaded.getvalue()
                )
                notes = [note for note in notes if note.get("entry")]
                if not notes:
                    st.error("No valid notes found in the upload.")
                else:
                    grouped: Dict[str, List[Dict[str, str]]] = {}
                    for note in notes:
                        hospital = note["hospital_number"]
                        grouped.setdefault(hospital, []).append(
                            {
                                "type": note["type"],
                                "entry": note["entry"],
                            }
                        )
                    inserted_total = 0
                    skipped_total = 0
                    for hospital, payload_notes in grouped.items():
                        inserted, skipped = add_doctor_notes(
                            DB_PATH, hospital, payload_notes
                        )
                        inserted_total += inserted
                        skipped_total += skipped
                    st.success(
                        f"Imported {inserted_total} notes. "
                        f"Skipped {skipped_total} notes."
                    )
                    st.rerun()

    with st.expander("Import Patient Documents", expanded=False):
        st.caption(
            "Upload CSV or JSON with columns: hospital_number, type, title, content."
        )
        uploaded_docs = st.file_uploader(
            "Upload documents",
            type=["csv", "json"],
            help="Rows without a matching hospital number are skipped.",
            key="docs_upload",
        )
        docs_submit = st.button("Import Documents")
        if docs_submit:
            if not uploaded_docs:
                st.error("Please upload a CSV or JSON file.")
            else:
                docs = parse_documents_upload(
                    uploaded_docs.name, uploaded_docs.getvalue()
                )
                docs = [doc for doc in docs if doc.get("content")]
                if not docs:
                    st.error("No valid documents found in the upload.")
                else:
                    grouped_docs: Dict[str, List[Dict[str, str]]] = {}
                    for doc in docs:
                        hospital = doc["hospital_number"]
                        grouped_docs.setdefault(hospital, []).append(
                            {
                                "type": doc["type"],
                                "title": doc["title"],
                                "content": doc["content"],
                            }
                        )
                    inserted_total = 0
                    skipped_total = 0
                    for hospital, payload_docs in grouped_docs.items():
                        inserted, skipped = add_documents(
                            DB_PATH, hospital, payload_docs
                        )
                        inserted_total += inserted
                        skipped_total += skipped
                    st.success(
                        f"Imported {inserted_total} documents. "
                        f"Skipped {skipped_total} documents."
                    )
                    st.rerun()

    with st.expander("OCR Import (Scanned PDFs or Images)", expanded=False):
        st.caption(
            "Upload a scanned PDF or image to extract text into patient documents."
        )
        ocr_type = st.radio(
            "Document type",
            options=["handwritten", "digital"],
            horizontal=True,
            key="ocr_doc_type",
        )
        ocr_title = st.text_input(
            "Document title",
            value="OCR Extract",
            key="ocr_title",
        )
        ocr_file = st.file_uploader(
            "Upload PDF or image",
            type=["pdf", "png", "jpg", "jpeg"],
            key="ocr_upload",
        )
        ocr_submit = st.button("Run OCR and Save")
        if ocr_submit:
            if not ocr_file:
                st.error("Please upload a PDF or image file.")
            elif pytesseract is None:
                st.error("Install pytesseract to enable OCR.")
            elif ocr_file.name.lower().endswith(".pdf") and convert_from_bytes is None:
                st.error("Install pdf2image to enable OCR for PDFs.")
            else:
                try:
                    extracted, confidence = ocr_from_upload(
                        ocr_file.name, ocr_file.getvalue()
                    )
                except Exception as exc:
                    st.error(f"OCR failed: {exc}")
                else:
                    if not extracted.strip():
                        st.warning("OCR produced no readable text.")
                    else:
                        inserted, skipped = add_documents(
                            DB_PATH,
                            hospital_number,
                            [
                                {
                                    "type": ocr_type,
                                    "title": ocr_title.strip()
                                    or "OCR Extract",
                                    "content": extracted.strip(),
                                    "ocr_confidence": confidence,
                                }
                            ],
                        )
                        if inserted:
                            if confidence is None:
                                st.success("OCR document saved.")
                            else:
                                st.success(
                                    "OCR document saved. "
                                    f"Confidence {confidence:.1f}%."
                                )
                            st.rerun()
                        else:
                            st.error("Unable to save OCR document.")

with right_col:
    st.subheader("Ask the Records Agent")
    st.caption(
        "Answers are grounded in the selected hospital number only."
    )
    question = st.text_area(
        "Ask about vitals, history, or clinical context",
        placeholder="Example: What did the last follow-up mention about blood pressure?",
        height=120,
    )
    submit = st.button("Generate Insight")
    if submit and question.strip():
        response = ask_agent(question, hospital_number)
        st.session_state.qa_history.append(
            {"question": question, "response": response}
        )

    for item in st.session_state.qa_history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["response"])
