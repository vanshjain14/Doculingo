import React, { useState, useEffect, useRef } from "react";
import "./Chat.css";

/* ----------------------------------------
   SAFE VOICE SELECTION (NO ERRORS)
----------------------------------------- */
function getBestVoice(preferredLang) {
  const voices = window.speechSynthesis.getVoices();
  if (!voices || voices.length === 0) return null;

  const validVoices = voices.filter(
    (v) => v.lang && typeof v.lang === "string"
  );

  let match = validVoices.find((v) =>
    v.lang.toLowerCase().startsWith(preferredLang)
  );
  if (match) return match;

  match = validVoices.find((v) => v.lang.toLowerCase().startsWith("en"));
  if (match) return match;

  return validVoices[0] || null;
}

/* ----------------------------------------
   SPEAK (WITH STOP + TOGGLE)
----------------------------------------- */
function speakText(text, language, setIsSpeaking) {
  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    return;
  }

  const utter = new SpeechSynthesisUtterance(text);
  const langCode = language === "de" ? "de" : language === "es" ? "es" : "en";
  const voice = getBestVoice(langCode);

  if (voice) utter.voice = voice;
  utter.rate = 1;
  utter.pitch = 1;

  utter.onend = () => setIsSpeaking(false);

  setIsSpeaking(true);
  window.speechSynthesis.speak(utter);
}

/* ----------------------------------------
   CHAT COMPONENT
----------------------------------------- */
function Chat({
  messages,
  isLoading,
  isSuggesting,
  submitQuery,
  suggestions,
  uiText,
  language,
}) {
  const [input, setInput] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState({});
  const chatWindowRef = useRef(null);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, isSuggesting]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading || isSuggesting) return;
    setInput("");
    submitQuery(input);
  };

  /* ----------------------------------------
     SEND FEEDBACK (👍 👎)
  ----------------------------------------- */
  const sendFeedback = async (index, rating) => {
    const botMessage = messages[index];
    const userMessage = messages[index - 1]?.text || "";

    setFeedbackStatus((prev) => ({ ...prev, [index]: "sending" }));

    try {
      await fetch("http://localhost:5000/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage,
          answer: botMessage.text,
          rating,
        }),
      });

      setFeedbackStatus((prev) => ({ ...prev, [index]: "done" }));

      setTimeout(() => {
        setFeedbackStatus((prev) => ({ ...prev, [index]: "thankyou" }));
      }, 1200);
    } catch (err) {
      console.error("Feedback error:", err);
      setFeedbackStatus((prev) => ({ ...prev, [index]: "error" }));
    }
  };

  return (
    <div className="chat-container">
      {/* CHAT WINDOW */}
      <div className="chat-window" ref={chatWindowRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <p>{msg.text}</p>

            {/* SPEAKER BUTTON (BOT MESSAGES ONLY) */}
            {msg.sender === "bot" && (
              <button
                className={`speak-btn ${isSpeaking ? "speaking" : ""}`}
                onClick={() => speakText(msg.text, language, setIsSpeaking)}
              >
                {isSpeaking ? "🔇" : "🔊"}
              </button>
            )}

            {/* FEEDBACK BUTTONS — ONLY WHEN NOT THINKING */}
            {msg.sender === "bot" &&
              !isLoading &&
              !isSuggesting &&
              !msg.text.includes("...") && (
                <div className="feedback-container">
                  {feedbackStatus[index] === "thankyou" ? (
                    <span className="feedback-thanks">
                      Thank you for the feedback!
                    </span>
                  ) : (
                    <div className="feedback-buttons">
                      <button
                        disabled={isLoading || isSuggesting}
                        onClick={() => sendFeedback(index, "positive")}
                        className={`fb-btn ${
                          feedbackStatus[index] === "done" ? "disabled" : ""
                        }`}
                      >
                        👍
                      </button>

                      <button
                        disabled={isLoading || isSuggesting}
                        onClick={() => sendFeedback(index, "negative")}
                        className={`fb-btn ${
                          feedbackStatus[index] === "done" ? "disabled" : ""
                        }`}
                      >
                        👎
                      </button>
                    </div>
                  )}
                </div>
              )}
          </div>
        ))}
      </div>

      {/* SUGGESTIONS */}
      <div className="suggestions-area">
        {isSuggesting && (
          <div className="suggestions-thinking">
            {uiText.suggesting}
            <span className="dot">.</span>
            <span className="dot">.</span>
            <span className="dot">.</span>
          </div>
        )}

        {!isSuggesting && suggestions.length > 0 && (
          <div className="suggestions-container">
            {suggestions.map((s, i) => (
              <button
                key={i}
                className="suggestion-button"
                onClick={() => submitQuery(s)}
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* INPUT BAR */}
      <form className="input-form" onSubmit={handleSend}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your document..."
          disabled={isLoading || isSuggesting}
        />
        <button type="submit" disabled={isLoading || isSuggesting}>
          {isLoading ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}

export default Chat;
