"""
FastAPI endpoint
Maintains session memory
"""



from fastapi import FastAPI
from pydantic import BaseModel
from graph import build_graph
from langchain_core.messages import HumanMessage
from fastapi.staticfiles import StaticFiles



app = FastAPI()

# Static files ko alag path par mount karo
app.mount("/static", StaticFiles(directory="static"), name="static")

graph = build_graph()

from fastapi.responses import FileResponse

@app.get("/")
def read_index():
    return FileResponse("static/index.html")


# In-memory session store (replace with Redis in production)
sessions = {}


class Query(BaseModel):
    session_id: str
    message: str


@app.post("/answer/")
def answer(query: Query):

    if query.session_id not in sessions:
        sessions[query.session_id] = {
            "session_id": query.session_id,
            "messages": [],
            "teacher_input": "",
            "intent": "",
            "extracted_topics": [],
            "topic_matches": [],
            "grouped_los": {},
            "selected_los": [],
            "chunk_ids": [],
            "chunk_contents": [],
            "generated_assessment": "",
            "last_assessment": "",
            "all_retrieved_los": [],
            "lo_page_index": 0,
            "stage": "start"
        }

    state = sessions[query.session_id]

    state["teacher_input"] = query.message
    state["messages"].append(HumanMessage(content=query.message))

    result = graph.invoke(state)

    sessions[query.session_id] = result

    return {
        "response": result["messages"][-1].content,
        "stage": result["stage"]
    }