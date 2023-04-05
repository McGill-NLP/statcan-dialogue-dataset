from datetime import datetime
import os
import json

import dash
from dash import dcc, html, Input, Output, State
import dash_auth
from dash import ALL
import dash_bootstrap_components as dbc
import pandas as pd

from eacl_code.models import CustomDPRModel
from eacl_code.utils import find_urls, get_data_dir


def is_copied(message):
    return " has accepted the chat.\n" in message and "\nVisitor " in message

def is_a_timestamp(line):
    if ":" not in line:
        return False
    
    time_split = line.split(":")
    if len(time_split) != 3:
        return False
    
    for unit in time_split:
        if len(unit) != 2 or not unit.isdigit():
            return False
    
    return True


def remove_blank_messages(conversation):
    return [c for c in conversation if c["text"] != ""]

def parse_pasted_conversation(text):
    lines = text.split("\n")
    out = {
        "agent": None,
        "visitor": None,
        "correctly_parsed": False,
    }

    # First, try to find the agent and visitor names
    for line in lines:
        if line.endswith(" has accepted the chat."):
            out["agent"] = line.replace("has accepted the chat.", "").strip()
        elif line.startswith("Visitor ") and out['visitor'] is None:
            out["visitor"] = line
    
        if out["agent"] and out["visitor"]:
            break
    
    if out['agent'] is None or out['visitor'] is None:
        return out

    # Start collecting conversation here
    actor = "visitor"
    conversation_data = [
        {"actor": actor, "text": "", "urls": []}
    ]

    for line in lines:
        if line.endswith(" has accepted the chat."):
            continue
    
        if line == out['agent']:
            actor = "agent"
            conversation_data.append({"actor": actor, "text": "", "urls": []})
        
        elif line == out['visitor']:
            actor = "visitor"
            conversation_data.append({"actor": actor, "text": "", "urls": []})

        elif is_a_timestamp(line):
            conversation_data.append({"actor": actor, "text": "", "urls": []})
    
        else:
            conversation_data[-1]["text"] += line + "\n"
            conversation_data[-1]["urls"] += find_urls(line)
    
    out['correctly_parsed'] = True
    out['conversation'] = remove_blank_messages(conversation_data)

    return out


def chatbox(text, box="agent"):
    style = {
        "max-width": "60%",
        "width": "max-content",
        "border-radius": 18,
        "margin-bottom": 5,
    }

    body_style = {
        "padding": "7px 12px",
    }

    if box == "agent":
        style["margin-left"] = "auto"
        style["margin-right"] = 0
        color = "primary"
        inverse = True

    elif box == "visitor":
        style["margin-left"] = 0
        style["margin-right"] = "auto"
        color = "light"
        inverse = False

    else:
        raise ValueError("Incorrect option for `box`.")

    return dbc.Card(
        dbc.CardBody(text, style=body_style), style=style, color=color, inverse=inverse
    )


def build_chatboxes(conversation):
    if type(conversation) == str:
        conversation = json.loads(conversation)

    chatboxes = []
    current_actor = None

    for c in conversation:
        if current_actor != c["actor"]:
            chatboxes.append(html.Div(style=dict(padding=5)))

        chatboxes.append(chatbox(c["text"], c["actor"]))

        current_actor = c["actor"]

    return chatboxes


def card_pre(header, body, pre_style=None):
    return dbc.Card(
        [
            dbc.CardHeader(header),
            dbc.CardBody(html.Pre(body, style=pre_style)),
        ]
    )


def was_triggered(component, prop):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False

    triggered_id, triggered_prop = ctx.triggered[0]["prop_id"].split(".", 1)
    return component.id == triggered_id and prop == triggered_prop


def indent_json(s, indent=2):
    return json.dumps(json.loads(s), indent=indent)


def format_pid(pid):
    pid = str(pid)
    pid = pid[:2] + "-" + pid[2:4] + "-" + pid[4:]
    return pid


def format_table_url(pid):
    return f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={pid}01"


def create_metadata_card(rank, pid, content):
    [[_, title], *content] = [x.split(":", 1) for x in content.split("\n")]

    title_div = html.H5(
        [f"{rank}. ", html.A(title, href=format_table_url(pid), target="_blank")]
    )

    feedback_form = html.Details([
        html.Summary("Add feedback"),
        dbc.Textarea(
            placeholder="Write feedback here, then click on 'Save' at the bottom of screen...",
            id={"name": "feedback-text", "pid": str(pid), "rank": rank},
            value="",
            rows=4,
        ),
    ])

    correct_answer_radio = dbc.RadioItems(
        id={"name": "correct-or-not-radio", "pid": str(pid), "rank": rank},
        options=[{"value": x, "label": x} for x in ["Correct", "Incorrect"]],
        inline=True,
    )

    content = [["Table", format_pid(pid)]] + content

    body = [
        title_div, 
        *[html.Div([html.I(left), ": ", right]) for left, right in content],
        correct_answer_radio,
        feedback_form,
    ]

    return dbc.Card(body, body=True, className="mb-3")


def retrieval_error_alert(e):
    return [
        dbc.Alert(
            f"There was an error processing your conversation. Please modify your conversation and trying again, and share this error message with the developer: {e}",
            color="danger",
        )
    ]

# ################################## VARIABLES ##################################
SAVE_DIR = "/home/toolkit/statcan-saved-chat/"
USE_WANDB = False

# ################################## COMPONENTS ##################################

message_input = dbc.Textarea(
    id="message-input",
    placeholder="Enter next message...",
    value="",
    rows=2,
)

submit_btn = dbc.Button("Submit", id="submit-btn", color="primary")

controls = dbc.Form(
    [
        dbc.Row(
            dbc.InputGroup(
                [
                    message_input,
                    submit_btn
                ]
            )
        ),
    ]
)

retrieval_loading = dbc.Spinner("Retrieve tables", id="retrieval-loading-spinner")

run_retrieval_btn = dbc.Button(
    retrieval_loading,
    id="run-retrieval-btn",
    color="secondary",
    size="sm",
)
remove_last_btn = dbc.Button(
    "Remove last",
    id="remove-last-btn",
    color="warning",
    size="sm",
)
clear_btn = dbc.Button(
    "Clear all",
    id="clear-btn",
    color="danger",
    size="sm",
)

save_btn = dbc.Button("Save results", id="save-btn", color="success", size="sm")

download_results = dcc.Download(id="download-results")

actor_radio = dbc.RadioItems(
    id="actor-radio",
    options=[{"value": x, "label": x} for x in ["auto", "visitor", "agent"]],
    value="auto",
    inline=True,
)

correctly_saved_alert = dbc.Alert(
    "Results correctly saved to server.",
    id="save-result-correct-alert",
    dismissable=True,
    is_open=False,
    color="success",
)

failed_to_save_alert = dbc.Alert(
    "Error saving results.",
    id="save-result-fail-alert",
    dismissable=True,
    is_open=False,
    color="danger",
)

conversation_bar = dbc.Form(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Actions"),
                    html.Br(),
                    dbc.ButtonGroup(
                        [
                            run_retrieval_btn,
                            save_btn,
                            remove_last_btn,
                            clear_btn,
                        ],
                    ),
                ],
                width="auto",
            ),
            dbc.Col(
                [
                    dbc.Label("Speaker"),
                    actor_radio,
                ],
                width="auto",
            ),
            dbc.Col([correctly_saved_alert, failed_to_save_alert], width="auto")
        ]
    )
)


conversation_store = dcc.Store(id="conversation-store", storage_type="session", data=[])

conversation_card_body = dbc.CardBody(
    id="conversation-card",
    style={"max-height": "fit-content", "overflow-y": "auto"},
)

conversation_card = dbc.Card(
    [
        dbc.CardHeader("Conversation"),
        conversation_card_body,
    ],
    style={"height": "calc(95vh - 200px)"},
)

retrieved_card_body = dbc.CardBody(
    id="retrieved-tables-card",
    style={"overflow-y": "auto"},
)

retrieval_card = dbc.Card(
    [
        dbc.CardHeader("Retrieved Tables"),
        retrieved_card_body,
    ],
    color="light",
    style={"height": "calc(95vh - 200px)"},
)


# ################################## APP ##################################
if USE_WANDB:
    import wandb
    wandb.init(project="saved-chats", entity="statscan-nlp", save_code=True)

os.makedirs(SAVE_DIR, exist_ok=True)


visitor_PASS_PAIRS = {"mila": "poplar-copartner"}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="DPR Table Retrieval Demo",
)
server = app.server
# auth = dash_auth.BasicAuth(app, visitor_PASS_PAIRS)


app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H2("DPR Table Retrieval Demo"),
        html.Br(),
        conversation_store,
        dbc.Row(
            [
                dbc.Col(
                    [
                        conversation_card,
                        html.Br(),
                        controls,
                    ],
                    width=6,
                ),
                dbc.Col([retrieval_card, html.Br(), conversation_bar], width=6),
            ]
        ),
        download_results
    ],
)

# ################################## MODEL/DATA LOADING ##################################
data_dir = get_data_dir()
model = CustomDPRModel(token=os.getenv("MCGILL_NLP_HUGGINGFACE_TOKEN"), q_enc_path="<path to question encoder>", ctx_enc_path="<path to context encoder>")
meta = pd.read_csv(data_dir / "retrieval/metadata.csv.zip").set_index("pid")

# ################################## CALLBACKS ##################################


@app.callback(
    Output(conversation_store.id, "data"),
    Output(message_input.id, "value"),
    Input(submit_btn.id, "n_clicks"),
    Input(message_input.id, "n_submit"),
    Input(clear_btn.id, "n_clicks"),
    Input(remove_last_btn.id, "n_clicks"),
    State(message_input.id, "value"),
    State(conversation_store.id, "data"),
    State(actor_radio.id, "value"),
)
def update_conversation_store(
    n_clicks,
    msg_input_click,
    __clear_btn,
    __remove_last_btn,
    message,
    conversation_data,
    actor,
):

    if was_triggered(remove_last_btn, "n_clicks"):
        return conversation_data[:-1], dash.no_update

    if was_triggered(clear_btn, "n_clicks"):
        return [], dash.no_update
    
    if (msg_input_click is None and n_clicks is None) or message == "":
        raise dash.exceptions.PreventUpdate

    if is_copied(message):
        parsed = parse_pasted_conversation(message)
        if parsed['correctly_parsed']:
            conversation_data += parsed['conversation']
        else:
            conversation_data.append(
                {
                    "actor": actor,
                    "message": message,
                    "text": "[SPECIAL MESSAGE] There was an error processing the message you pasted."
                }
            )
    else:
        if actor == "auto":
            if len(conversation_data) == 0:
                actor = "visitor"
            elif conversation_data[-1]["actor"] == "visitor":
                actor = "agent"
            else:
                actor = "visitor"

        conversation_data.append(
            {"actor": actor, "text": message.strip(), "urls": find_urls(message)}
        )

    return conversation_data, ""


@app.callback(
    Output(conversation_card_body.id, "children"),
    Input(conversation_store.id, "data"),
)
def update_conversation_box(conversation_data):
    return build_chatboxes(conversation_data)


@app.callback(
    Output(retrieved_card_body.id, "children"),
    Output(retrieval_loading.id, "children"),
    Input(run_retrieval_btn.id, "n_clicks"),
    Input(remove_last_btn.id, "n_clicks"),
    Input(clear_btn.id, "n_clicks"),
    State(conversation_store.id, "data"),
)
def run_retrieval(n_clicks, __remove_last_btn_n_clicks, __clear_btn_n_clicks, conversation_data):
    if n_clicks is None or len(conversation_data) == 0:
        raise dash.exceptions.PreventUpdate

    if was_triggered(run_retrieval_btn, "n_clicks"):
        try:
            sorted_indices = model.retrieve_table_indices(conversation_data)
        except Exception as e:
            return retrieval_error_alert(e), dash.no_update

        retrieved_pids = meta.iloc[sorted_indices].index.values

        cards = [
            create_metadata_card(i, pid, meta.loc[pid, "basic_info"])
            for i, pid in enumerate(retrieved_pids, 1)
        ]

        return cards, dash.no_update
    
    if was_triggered(clear_btn, "n_clicks") or was_triggered(remove_last_btn, "n_clicks"):
        return [], dash.no_update


@app.callback(
    Output(correctly_saved_alert.id, "is_open"),
    Output(failed_to_save_alert.id, "is_open"),
    Output(download_results.id, "data"),
    Input(save_btn.id, "n_clicks"),
    State(conversation_store.id, "data"),
    State({"name": "correct-or-not-radio", "pid": ALL, "rank": ALL}, "id"),
    State({"name": "feedback-text", "pid": ALL, "rank": ALL}, "id"),
    State({"name": "correct-or-not-radio", "pid": ALL, "rank": ALL}, "value"),
    State({"name": "feedback-text", "pid": ALL, "rank": ALL}, "value"),
)
def save_results(n_clicks, conversation_data, correct_or_not_id, feedback_text_id, correct_or_not, feedback_text):
    if n_clicks is None or len(conversation_data) == 0:
        raise dash.exceptions.PreventUpdate

    if (
        any(x['pid'] != y['pid'] for x, y in zip(correct_or_not_id, feedback_text_id))
    ) or (
        any(x['rank'] != y['rank'] for x, y in zip(correct_or_not_id, feedback_text_id))
    ):
        return False, True, dash.no_update
    
    retrieval_feedback = []
    for i in range(len(correct_or_not_id)):
        di = {
            'correct_or_not': correct_or_not[i],
            'feedback': feedback_text[i],
            'pid': correct_or_not_id[i]['pid'],
            'rank': correct_or_not_id[i]['rank'],
        }
        retrieval_feedback.append(di)
    
    output = {
        'conversation': conversation_data,
        'retrieval_feedback': retrieval_feedback,
    }

    filename = f'{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}.json'

    try:
        with open(os.path.join(SAVE_DIR, filename), 'w') as f:
            json.dump(output, f, indent=4)

        if USE_WANDB:
            wandb.log(output)
    
    except Exception as e:
        return False, True, dash.no_update
    
    return True, False, dict(content=json.dumps(output, indent=4), filename=filename)

if __name__ == "__main__":
    app.run_server(debug=True, port=8081, host="0.0.0.0")
