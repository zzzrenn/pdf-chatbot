import base64
import os
import shutil
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import requests
from dash import Input, Output, State, callback, ctx, dcc, html
from dotenv import load_dotenv
from utils.logger import setup_logger

# Environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

# Setup logger
logger = setup_logger("frontend")


class DocumentManager:
    def __init__(self, api_url: str = "http://localhost:8000"):
        logger.info(f"DocumentManager initialized with backend API URL: {api_url}")
        self.api_url = api_url

    def upload_document(self, file) -> bool:
        """Upload a document to the backend"""
        logger.info(f"Attempting to upload document: {file.name}")
        try:
            files = {"file": (file.name, file, "application/pdf")}
            response = requests.post(f"{self.api_url}/upload", files=files)
            success = response.status_code == 200
            if success:
                logger.info(f"Successfully uploaded document: {file.name}")
            else:
                logger.error(
                    f"Failed to upload document: {file.name}, Status: {response.status_code}"
                )
            return success
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            return False

    def get_documents(self) -> List[Dict]:
        """Fetch list of documents from backend"""
        logger.debug("Fetching document list")
        response = requests.get(f"{self.api_url}/documents")
        if response.status_code == 200:
            documents = response.json()
            logger.info(f"Successfully fetched {len(documents)} documents")
            return documents
        logger.warning(f"Failed to fetch documents, Status: {response.status_code}")
        return []

    def get_document_content(self, filename: str) -> bytes:
        """Fetch document content from backend"""
        logger.info(f"Fetching content for document: {filename}")
        try:
            response = requests.get(f"{self.api_url}/document/{filename}")
            if response.status_code == 200:
                logger.info(f"Successfully fetched content for: {filename}")
                return response.content
            logger.warning(f"Failed to fetch document content, Status: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching document content: {str(e)}")
            return None


class ChatInterface:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        logger.info(f"ChatInterface initialized with backend API URL: {api_url}")

    def get_response(self, question: str) -> Dict:
        """Get response from chatbot"""
        logger.info(f"Getting response for question: {question}")
        try:
            response = requests.post(f"{self.api_url}/chat", json={"question": question})
            if response.status_code == 200:
                logger.info("Successfully received chat response")
                return response.json()
            logger.warning(f"Failed to get chat response, Status: {response.status_code}")
            return {
                "answer": "Error: Could not get response",
                "source_documents": [],
            }
        except Exception as e:
            logger.error(f"Error getting chat response: {str(e)}")
            return {
                "answer": "Error: Could not get response",
                "source_documents": [],
            }


# Initialize Dash app with dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
doc_manager = DocumentManager(api_url=BACKEND_URL)
chat_interface = ChatInterface(api_url=BACKEND_URL)

# Layout
app.layout = dbc.Container(
    [
        # Store components for managing state
        dcc.Store(id="current-pdf-content"),
        dcc.Store(id="active-tab", data="chat"),
        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        html.H3("Document Management", className="mb-4"),
                        dcc.Upload(
                            id="upload-document",
                            children=html.Div(["Drag and Drop or ", html.A("Select PDF File")]),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px 0",
                            },
                            multiple=False,
                            accept="application/pdf",
                        ),
                        dbc.Spinner(
                            html.Div(id="upload-status"), color="primary", type="border", size="sm"
                        ),
                        html.H4("Document List", className="mt-4 mb-3"),
                        dbc.Button(
                            [html.I(className="fas fa-sync-alt me-2"), "Refresh List"],
                            id="refresh-button",
                            color="primary",
                            className="mb-3",
                        ),
                        html.Div(
                            id="document-list",
                            style={"maxHeight": "calc(100vh - 300px)", "overflowY": "auto"},
                        ),
                    ],
                    width=3,
                    className="bg-dark p-4",
                    style={"height": "100vh"},
                ),
                # Main content
                dbc.Col(
                    [
                        dcc.Tabs(
                            [
                                # Chat Interface Tab
                                dcc.Tab(
                                    label="Chat Interface",
                                    value="chat",
                                    children=[
                                        html.Div(
                                            [
                                                html.Div(
                                                    id="chat-messages",
                                                    style={
                                                        "height": "calc(100vh - 200px)",
                                                        "overflowY": "auto",
                                                        "padding": "20px",
                                                    },
                                                ),
                                                dbc.InputGroup(
                                                    [
                                                        dbc.Input(
                                                            id="chat-input",
                                                            type="text",
                                                            placeholder="Ask a question...",
                                                            style={
                                                                "backgroundColor": "#2c3e50",
                                                                "color": "white",
                                                            },
                                                        ),
                                                        dbc.Button(
                                                            "Send",
                                                            id="send-button",
                                                            color="primary",
                                                        ),
                                                    ],
                                                    className="mt-3",
                                                ),
                                            ]
                                        )
                                    ],
                                    style={"backgroundColor": "#1a1a1a"},
                                    selected_style={"backgroundColor": "#2c3e50"},
                                ),
                                # PDF Viewer Tab
                                dcc.Tab(
                                    label="PDF Viewer",
                                    value="pdf",
                                    children=[
                                        dbc.Spinner(
                                            html.Div(
                                                id="pdf-viewer",
                                                style={"height": "calc(100vh - 150px)"},
                                            ),
                                            color="primary",
                                            type="border",
                                        )
                                    ],
                                    style={"backgroundColor": "#1a1a1a"},
                                    selected_style={"backgroundColor": "#2c3e50"},
                                ),
                            ],
                            id="tabs",
                            value="chat",
                        )
                    ],
                    width=9,
                    style={"height": "100vh", "paddingLeft": 0},
                ),
            ],
            className="g-0",
        ),  # Remove gutters for better spacing
    ],
    fluid=True,
    style={"backgroundColor": "#1a1a1a", "minHeight": "100vh", "padding": 0},
)


# Callbacks
@callback(
    Output("document-list", "children"),
    [Input("refresh-button", "n_clicks")],
    prevent_initial_call=False,
)
def update_document_list(_):
    documents = doc_manager.get_documents()
    return dbc.Table(
        [
            html.Thead([html.Tr([html.Th("Document Name", style={"width": "100%"})])]),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                html.A(
                                    doc["filename"],
                                    id={"type": "doc-link", "index": doc["filename"]},
                                    className="text-light text-decoration-none",
                                    style={
                                        "cursor": "pointer",
                                        "whiteSpace": "nowrap",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "display": "block",
                                    },
                                )
                            )
                        ]
                    )
                    for doc in documents
                ]
            ),
        ],
        bordered=True,
        dark=True,
        hover=True,
        responsive=True,
        size="sm",
    )


@callback(
    [Output("pdf-viewer", "children"), Output("tabs", "value")],
    [
        Input({"type": "doc-link", "index": dash.ALL}, "n_clicks"),
        Input({"type": "source-link", "index": dash.ALL}, "n_clicks"),
    ],
    # [State({"type": "doc-link", "index": dash.ALL}, "id"),
    #  State({"type": "source-link", "index": dash.ALL}, "id")],
    prevent_initial_call=True,
)
def update_pdf_viewer(doc_clicks, source_clicks):
    if not ctx.triggered_id or not ctx.triggered[0]["value"]:
        return "Select a document to view", "chat"

    triggered = ctx.triggered_id
    filename = None
    page = None

    if triggered.get("type") == "doc-link":
        filename = triggered.get("index")
    elif triggered.get("type") == "source-link":
        # Parse source string (format: "filename:page")
        source_str = triggered.get("index")
        if ":" in source_str:
            filename, page = source_str.split(":")

    if filename:
        pdf_content = doc_manager.get_document_content(filename)
        if pdf_content:
            base64_pdf = base64.b64encode(pdf_content).decode("utf-8")
            viewer = html.Iframe(
                src=f"data:application/pdf;base64,{base64_pdf}#page={page if page else 0}",
                style={"width": "100%", "height": "100%", "border": "none"},
            )
            return viewer, "pdf"

    return "Error loading PDF", "pdf"


@callback(
    Output("upload-status", "children"),
    Input("upload-document", "contents"),
    State("upload-document", "filename"),
)
def upload_document(contents, filename):
    if contents is None:
        return ""

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, filename)
    logger.debug(f"Saving file to: {file_path}")
    with open(file_path, "wb") as f:
        f.write(decoded)
    logger.info(f"File saved successfully: {filename}")

    with open(file_path, "rb") as f:
        success = doc_manager.upload_document(f)

    if success:
        # Move documents from upload dir to document dir
        logger.debug("Moving processed document to document storage directory")
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            shutil.move(file_path, DOCUMENT_DIR)
        shutil.rmtree(UPLOAD_DIR)
        logger.info("Document moved to storage directory succesfully")
        return html.Div("Upload successful!", style={"color": "#2ecc71"})
    return html.Div("Upload failed!", style={"color": "#e74c3c"})


@callback(
    [Output("chat-messages", "children"), Output("chat-input", "value")],
    Input("send-button", "n_clicks"),
    [State("chat-input", "value"), State("chat-messages", "children")],
    prevent_initial_call=True,
)
def update_chat(n_clicks, input_value, current_messages):
    if not input_value:
        return current_messages or [], ""

    current_messages = current_messages or []

    # Add user message
    user_message = dbc.Card(
        dbc.CardBody(input_value),
        className="mb-2 ml-auto",
        style={
            "width": "70%",
            "margin-left": "30%",
            "backgroundColor": "#2c3e50",
            "color": "white",
        },
    )

    # Get bot response
    response = chat_interface.get_response(input_value)

    # Create source links
    source_links = []
    for filename, page in response.get("source_documents", []):
        filename = filename.split(os.sep)[-1]
        page_num = int(page.split(": ")[-1])
        source_links.append(
            html.A(
                f"{filename} (page {page_num})",
                id={"type": "source-link", "index": f"{filename}:{page_num}"},
                className="mr-2 text-info",
                style={"cursor": "pointer", "marginRight": "10px"},
                n_clicks=0,
            )
        )

    # Add bot message with markdown support and clickable sources
    bot_message = dbc.Card(
        [
            dbc.CardBody(
                [
                    # Use dcc.Markdown instead of html.P for markdown rendering
                    dcc.Markdown(
                        response["answer"],
                        # Add styling for markdown content
                        style={
                            "backgroundColor": "#34495e",
                            "padding": "10px",
                            "borderRadius": "5px",
                        },
                    ),
                    html.Small([html.P(["Sources: "] + source_links)], className="text-muted"),
                ]
            )
        ],
        className="mb-2",
        style={"width": "70%", "backgroundColor": "#34495e", "color": "white"},
    )

    return current_messages + [user_message, bot_message], ""


if __name__ == "__main__":
    app.run_server(debug=True)
