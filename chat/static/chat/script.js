// Clear previous cache/searches on page load
localStorage.removeItem('smartstyle_session');
localStorage.removeItem('user_gender');

let sessionId = localStorage.getItem('smartstyle_session') || generateSessionId();
let isDarkMode = localStorage.getItem('darkMode') === 'true';

// Initialize theme
if (isDarkMode) {
    document.body.classList.add('dark');
    document.getElementById('themeIcon').textContent = '☀️';
}

function generateSessionId() {
    const id = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('smartstyle_session', id);
    return id;
}

function toggleTheme() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark');
    document.getElementById('themeIcon').textContent = isDarkMode ? '☀️' : '🌙';
    localStorage.setItem('darkMode', isDarkMode);
}

let userGender = localStorage.getItem('user_gender') || '';

function sendMessage(message) {
    if (!message.trim()) return;

    // Add user message
    addMessage(message, 'user');

    // Show typing indicator
    showTyping();

    // Handle gender selection if not set
    if (!userGender) {
        const gender = message.toLowerCase().trim();
        if (gender.includes('men') || gender.includes('male')) {
            userGender = 'men';
        } else if (gender.includes('woman') || gender.includes('female')) {
            userGender = 'woman';
        } else if (gender.includes('both')) {
            userGender = 'both';
        } else {
            // No gender specified, proceed without setting gender
            userGender = 'both'; // Default to both if not specified
        }

        localStorage.setItem('user_gender', userGender);

        // Update preferences
        fetch('/api/preferences/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, gender: userGender })
        });

        // Don't show confirmation message, just proceed to normal chat
    }

    // Send to API
    fetch('/api/chat/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            session_id: sessionId
        })
    })
    .then(response => response.json())
    .then(data => {
        hideTyping();
        if (data.error) {
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        } else {
            addMessage(data.response, 'bot');
            sessionId = data.session_id;
            localStorage.setItem('smartstyle_session', sessionId);
        }
    })
    .catch(error => {
        hideTyping();
        addMessage('Sorry, I\'m having trouble connecting. Please check your connection and try again.', 'bot');
        console.error('Error:', error);
    });

    // Clear input
    document.getElementById('messageInput').value = '';
}

function addMessage(content, type) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const avatarClass = type === 'bot' ? 'bot' : 'user';
    const avatarImg = type === 'bot' ? '/static/chat/chatbot.png' : '/static/chat/user.png';

    // Format content for bot messages
    let formattedContent = content;
    if (type === 'bot') {
        formattedContent = formatBotMessage(content);
    }

    messageDiv.innerHTML = `
        <div class="message-avatar ${avatarClass}"><img src="${avatarImg}" alt="${type}"></div>
        <div class="message-content">${formattedContent}</div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function formatBotMessage(content) {
    // Convert bullet points to HTML lists
    let formatted = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold text
        .replace(/•\s*(.*?)(?=\n|$)/g, '<li>$1</li>') // Bullet points
        .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>'); // Wrap in ul

    // Clean up multiple consecutive li tags
    formatted = formatted.replace(/(<\/li>\s*<li>)/g, '</li><li>');

    // Convert line breaks to <br> for better formatting
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

function showTyping() {
    document.getElementById('typingIndicator').style.display = 'block';
}

function hideTyping() {
    document.getElementById('typingIndicator').style.display = 'none';
}

function sendQuickReply(message) {
    // Function kept for potential future use
    document.getElementById('messageInput').value = message;
    sendMessage(message);
}

// Event listeners
document.getElementById('sendButton').addEventListener('click', () => {
    const message = document.getElementById('messageInput').value.trim();
    if (message) {
        sendMessage(message);
    }
});

document.getElementById('messageInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const message = document.getElementById('messageInput').value.trim();
        if (message) {
            sendMessage(message);
        }
    }
});

document.getElementById('themeToggle').addEventListener('click', toggleTheme);

// Quick reply buttons - using onclick attributes from HTML

// Auto-focus input
document.getElementById('messageInput').focus();
