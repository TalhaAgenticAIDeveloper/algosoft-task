// Generate random session ID
let session_id = localStorage.getItem("session_id");

if (!session_id) {
    session_id = crypto.randomUUID();
    localStorage.setItem("session_id", session_id);
}

const chatBox = document.getElementById("chat-box");
const inputField = document.getElementById("user-input");
const stageDisplay = document.getElementById("stage-display");

function addMessage(content, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message");
    messageDiv.classList.add(sender);
    messageDiv.innerText = content;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {

    const message = inputField.value.trim();
    if (!message) return;

    addMessage(message, "user");
    inputField.value = "";

    try {
        const response = await fetch("/answer/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: session_id,
                message: message
            })
        });

        const data = await response.json();

        addMessage(data.response, "bot");
        stageDisplay.innerText = "Current Stage: " + data.stage;

    } catch (error) {
        addMessage("Error connecting to server.", "bot");
        console.error(error);
    }
}

// Send on Enter key
inputField.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});