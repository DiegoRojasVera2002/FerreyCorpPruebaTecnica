// static/js/script.js
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function addMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const icon = document.createElement('i');
    icon.className = isUser ? 'fas fa-user' : 'fas fa-robot';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = message;
    
    if (isUser) {
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(icon);
    } else {
        messageDiv.appendChild(icon);
        messageDiv.appendChild(contentDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    
    // Agregar mensaje del usuario
    addMessage(message, true);
    userInput.value = '';
    
    // Mostrar indicador de carga
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message';
    loadingDiv.innerHTML = '<i class="fas fa-robot"></i><div class="message-content">Pensando...</div>';
    chatMessages.appendChild(loadingDiv);
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: message }),
        });
        
        const data = await response.json();
        
        // Remover indicador de carga
        chatMessages.removeChild(loadingDiv);
        
        // Agregar respuesta del bot
        addMessage(data.response);
        
    } catch (error) {
        // Remover indicador de carga
        chatMessages.removeChild(loadingDiv);
        
        // Mostrar mensaje de error
        addMessage('Lo siento, hubo un error al procesar tu pregunta. Por favor, intenta de nuevo.');
    }
}

// Enfocar el input al cargar la página
window.onload = () => {
    userInput.focus();
};