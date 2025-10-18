import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
from PIL import Image
import io
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny
import json
from ultralytics import YOLO  # for second model
import gdown
import os

# ------------------- Utility for downloading models -------------------
def download_if_missing(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"⏬ Downloading model from {url} ...")
        gdown.download(url, dest_path, quiet=False)
        print(f"✅ Download complete: {dest_path}")

# ------------------- Paths & URLs -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "benthic_artifacts"), exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "benthic_artifacts", "convnext_tiny_best.pth")
SECOND_MODEL_PATH = os.path.join(BASE_DIR, "benthic_artifacts", "benthic_yolov8_best.pt")
CLASSES_PATH = os.path.join(BASE_DIR, "benthic_artifacts", "classes.json")

MODEL_URL = "https://drive.google.com/uc?export=download&id=153xQQUIGXJtt0H6wqn-fanxVGh5eZ3y7"
SECOND_MODEL_URL = "https://drive.google.com/uc?export=download&id=11nygQYV9QyfXG2hcPgTIIGgH-My8GG7o"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Lazy-loading Models -------------------
model = None
second_model = None
CLASS_NAMES = None

def get_models():
    """Load models only on first use (lazy initialization)."""
    global model, second_model, CLASS_NAMES

    if model is None or second_model is None:
        print("⏳ Loading models for the first time...")

        # Ensure models are downloaded
        download_if_missing(MODEL_URL, MODEL_PATH)
        download_if_missing(SECOND_MODEL_URL, SECOND_MODEL_PATH)

        # Load class labels
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)

        # Load ConvNeXt model
        model = convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()

        # Load YOLO model
        second_model = YOLO(SECOND_MODEL_PATH)

        print("✅ Models loaded successfully!")

    return model, second_model, CLASS_NAMES

# ------------------- Image Transform -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------- Utility Functions -------------------
def classify_image(image_array):
    model, _, CLASS_NAMES = get_models()
    image_pil = Image.fromarray(image_array.astype('uint8'), mode="RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_index = outputs.argmax(dim=1).item()
        return CLASS_NAMES[pred_index]

def draw_box_on_image(image_array):
    _, second_model, _ = get_models()
    image_pil = Image.fromarray(image_array.astype('uint8'), mode="RGB")
    results = second_model(image_pil)
    boxed_image = results[0].plot()  # numpy array with boxes
    buffered = io.BytesIO()
    Image.fromarray(boxed_image).save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# ------------------- Dash Setup -------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
image_store = []

# ------------------- Layout Components -------------------
banner = html.Div([
    html.Img(src='/assets/wm.png', className='banner-image'),
    html.Div("Benthic Species Recognition", className='banner-title'),
    html.Img(src='/assets/wm.png', className='banner-image')
], className='banner')

description_box = html.Div([
    html.H4("Description"),
    html.P(
        "This is an AI-powered solution for benthic species identification in marine science research and monitoring. "
        "We include a robust classification model trained on 10,500 underwater images to accurately identify seven benthic species "
        "(Scallop, Roundfish, Crab, Whelk, Skate, Flatfish, and Eel). "
        "A second object detection model can locate multiple organisms within complex seafloor scenes."
    )
], className='info-box')

instruction_box = html.Div([
    html.H4("Instructions"),
    html.P(
        "Attach your underwater image below. After uploading, you’ll see both the original image and the detection output. "
        "The classification result will also appear. You can visit the Gallery to review all previously uploaded images "
        "with filenames, timestamps, and detection results."
    )
], className='info-box')

box_container = html.Div(
    [description_box, instruction_box],
    style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'marginTop': '20px'}
)

# ------------------- Page Layouts -------------------
def upload_page():
    return html.Div([
        banner,
        box_container,
        html.Div([
            dbc.Container([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        html.Span('Drag and Drop or ', className='upload-text'),
                        html.A('Select an Image', className='upload-text')
                    ]),
                    className='upload-box',
                    accept='image/*',
                    multiple=True
                ),
                dcc.Loading(
                    id="loading-output",
                    type="circle",
                    color="#004080",
                    children=html.Div(id='output-image-upload')
                ),
                html.Br(),
                html.Div(
                    dbc.Button("Go to Gallery", href="/gallery", color="secondary", style={
                        'marginTop': '20px',
                        'marginBottom': '30px'
                    }),
                    style={'textAlign': 'center'}
                )
            ])
        ], style={'marginTop': '30px'}),
    ], className='center-box-wrapper')

def gallery_page():
    if not image_store:
        return html.Div([
            banner,
            dbc.Container([
                html.H3("Gallery of Uploaded Images", className='text-center', style={
                    'color': 'white', 'fontFamily': 'Times New Roman, serif',
                    'marginTop': '50px', 'fontSize': '30px'
                }),
                html.P("No images uploaded yet.", style={'textAlign': 'center', 'color': 'white'}),
                dbc.Button("Back to Upload", href="/", color="secondary", className='mt-3')
            ])
        ])

    gallery_items = []
    for item in image_store:
        gallery_items.append(html.Div([
            html.H6(item.get('filename', 'Unknown'), style={'fontWeight': 'bold', 'color': '#003366'}),
            html.P(f"Prediction: {item['prediction']}"),
            html.P(f"Uploaded: {item['timestamp']}", style={'fontSize': '12px', 'color': '#666'}),
            html.Img(src=item['original']),
            html.Img(src=item['boxed'], style={'marginTop': '10px'})
        ], className='gallery-item'))

    return html.Div([
        banner,
        dbc.Container([
            html.H3("Gallery of Uploaded Images", className='text-center', style={
                'color': 'white', 'fontFamily': 'Times New Roman, serif',
                'marginTop': '50px', 'fontSize': '30px'
            }),
            dbc.Button("Back to Upload", href="/", color="secondary", className='mb-4'),
            html.Div(gallery_items, style={
                'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'gap': '20px'
            })
        ])
    ])

# ------------------- App Routing -------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/gallery':
        return gallery_page()
    else:
        return upload_page()

# ------------------- Upload Callback -------------------
@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
)
def update_output(contents, filenames):
    if contents is None:
        return

    if not isinstance(contents, list):
        contents = [contents]
        filenames = [filenames]

    children = []

    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        try:
            image = Image.open(io.BytesIO(decoded)).convert("RGB")
            image_array = np.array(image)

            prediction = classify_image(image_array)
            boxed_image_b64 = draw_box_on_image(image_array)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            image_store.append({
                'original': content,
                'boxed': boxed_image_b64,
                'prediction': prediction,
                'timestamp': timestamp,
                'filename': filename
            })

            children.append(html.Div([
                html.H5(f'Prediction: {prediction}', style={'color': '#003366', 'fontWeight': 'bold'}),
                html.Div([
                    html.Img(src=content, style={'maxWidth': '220px', 'borderRadius': '8px'}),
                    html.Img(src=boxed_image_b64, style={'maxWidth': '220px', 'borderRadius': '8px', 'marginLeft': '20px'})
                ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px'}),
                html.P(f"Uploaded: {timestamp}", style={'fontSize': '13px', 'color': '#555'}),
                html.Hr()
            ], className='prediction-card'))

        except Exception as e:
            children.append(html.Div([html.P(f"Error processing {filename}: {str(e)}")]))
    return children

# ------------------- Run App -------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Render assigns PORT dynamically
    app.run(host='0.0.0.0', port=port, debug=False)
