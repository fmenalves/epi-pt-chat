// Array to store chat history
const chatHistory = [];

async function sendChat() {
    event.preventDefault();

    const medicamentoSelect = document.getElementById("medicamentoSelect");
    const selectedMedicamento = medicamentoSelect.options[medicamentoSelect.selectedIndex].value;
    const selectedDosagem = medicamentoSelect.options[medicamentoSelect.selectedIndex].getAttribute("dosagem");

    const userInput = document.getElementById("userInput").value;

    if (!userInput.trim()) {
        alert("Por favor, insira uma mensagem.");
        return;
    }

    // Add user's message to the history
    chatHistory.push({ sender: 'user', message: userInput });
    appendMessage(userInput, null, 'user-message'); // User message doesn't have additional info

    // Prepare the payload with the last 5 messages
    const recentHistory = chatHistory.slice(-5); // Get the last 5 messages
    const data = {
        medicamento: selectedMedicamento,
        dosagem: selectedDosagem,
        recent_history: recentHistory
    };

    try {
        const response = await fetch('/demochat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Add bot's response to the history
        chatHistory.push({ sender: 'bot', message: result.response });
        appendMessage(result.response, result.additional_info, 'bot-message'); // Pass additional info for bot response
    } catch (error) {
        console.error("Error:", error);
        alert("Ocorreu um erro ao enviar a mensagem. Por favor, tente novamente.");
    }

    document.getElementById("userInput").value = "";
}

function appendMessage(message, additionalInfo, className) {
    const chatArea = document.getElementById("chatArea");
    const messageDiv = document.createElement("div");

    // Set message text and class
    messageDiv.textContent = message;
    messageDiv.classList.add('message', className);

    // Add additional info as a tooltip if available
    if (additionalInfo) {
        messageDiv.setAttribute('title', additionalInfo); // For default browser tooltip
        messageDiv.setAttribute('data-tooltip', additionalInfo); // For custom tooltip (CSS)
    }

    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}
