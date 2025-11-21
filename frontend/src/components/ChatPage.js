// src/components/ChatPage.js

import React, { useState } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

import FileUpload from "./FileUpload";
import Chat from "./Chat";
import LanguageSelector from "./LanguageSelector";

const translations = {
  en: {
    thinking: "Thinking...",
    summarizing: "Generating summary...",
    suggesting: "Generating suggestions...",
    fileProcessed: "File processed! You can now ask questions.",
    summarizeButton: "Summarize Document",
    exportButton: "Export to PDF",
  },
  es: {
    thinking: "Pensando...",
    summarizing: "Generando resumen...",
    suggesting: "Generando sugerencias...",
    fileProcessed: "¡Archivo procesado! Ya puedes hacer preguntas.",
    summarizeButton: "Resumir documento",
    exportButton: "Exportar a PDF",
  },
  de: {
    thinking: "Nachdenken...",
    summarizing: "Zusammenfassung wird erstellt...",
    suggesting: "Vorschläge werden generiert...",
    fileProcessed: "Datei verarbeitet! Sie können jetzt Fragen stellen.",
    summarizeButton: "Dokument zusammenfassen",
    exportButton: "Als PDF exportieren",
  },
};

function ChatPage() {
  const [isFileUploaded, setIsFileUploaded] = useState(false);
  const [language, setLanguage] = useState("en");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestions, setSuggestions] = useState([]);

  const uiText = translations[language];

  // ------------------------------------------------------------------
  // SEND QUESTION
  // ------------------------------------------------------------------
  const submitQuery = async (queryText) => {
    if (!queryText.trim()) return;

    setMessages((prev) => [
      ...prev,
      { text: queryText, sender: "user" },
      { text: uiText.thinking, sender: "bot" },
    ]);

    setIsLoading(true);
    setSuggestions([]);
    setIsSuggesting(false);

    try {
      const response = await fetch("http://localhost:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: queryText, language }),
      });

      const data = await response.json();
      const answer = data.answer || "Sorry, I could not generate an answer.";

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { text: answer, sender: "bot" };
        return updated;
      });

      setIsLoading(false);

      // --- GET SUGGESTIONS ---
      setIsSuggesting(true);

      const sugRes = await fetch("http://localhost:5000/suggest-questions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: queryText,
          language,
        }),
      });

      const sugData = await sugRes.json();

      const cleanSuggestions = (sugData.suggestions || [])
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

      setSuggestions(cleanSuggestions);
    } catch (error) {
      console.error("Query error:", error);
    } finally {
      setIsSuggesting(false);
    }
  };

  // ------------------------------------------------------------------
  // SUMMARIZE (SUGGESTIONS WILL BE CLEARED — FIXED)
  // ------------------------------------------------------------------
  const handleSummarize = async () => {
    setMessages((prev) => [
      ...prev,
      { text: uiText.summarizing, sender: "bot" },
    ]);

    setSuggestions([]); // <-- FIX: Remove old suggestions
    setIsSuggesting(false);

    setIsLoading(true);

    const res = await fetch("http://localhost:5000/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language }),
    });

    const data = await res.json();

    setMessages((prev) =>
      prev.map((msg, i) =>
        i === prev.length - 1 ? { text: data.summary, sender: "bot" } : msg
      )
    );

    setIsLoading(false);
  };

  const handleExport = async () => {
    const chatWindow = document.querySelector(".chat-window");
    if (!chatWindow) return;

    // Apply clean-print mode
    chatWindow.classList.add("print-clean");

    // Save original size
    const oldHeight = chatWindow.style.height;
    const oldOverflow = chatWindow.style.overflow;

    try {
      chatWindow.style.height = "auto";
      chatWindow.style.overflow = "visible";

      await new Promise((res) => setTimeout(res, 200));

      const canvas = await html2canvas(chatWindow, {
        scale: 2,
        useCORS: true,
        backgroundColor: "hsla(0, 0%, 0%, 1.00)", // IMPORTANT
      });

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();

      const imgHeight = (canvas.height * pageWidth) / canvas.width;
      let heightLeft = imgHeight;
      let position = 0;

      pdf.addImage(imgData, "PNG", 0, position, pageWidth, imgHeight);
      heightLeft -= pageHeight;

      while (heightLeft > 0) {
        pdf.addPage();
        position = heightLeft - imgHeight;
        pdf.addImage(imgData, "PNG", 0, position, pageWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      pdf.save("DocuLingo_Chat.pdf");
    } catch (err) {
      console.error("PDF export error:", err);
    } finally {
      // Remove forced light mode
      chatWindow.classList.remove("print-clean");

      chatWindow.style.height = oldHeight;
      chatWindow.style.overflow = oldOverflow;
    }
  };

  return (
    <div className="chat-page">
      <header className="App-header">
        <h1>DocuLingo AI</h1>

        {isFileUploaded && (
          <div className="header-controls">
            <button
              onClick={
                !isLoading && !isSuggesting ? handleSummarize : undefined
              }
              disabled={isLoading || isSuggesting}
              className="header-button"
            >
              {uiText.summarizeButton}
            </button>

            <button
              onClick={handleExport}
              disabled={isLoading}
              className="header-button"
            >
              {uiText.exportButton}
            </button>

            <LanguageSelector
              language={language}
              onLanguageChange={setLanguage}
            />
          </div>
        )}
      </header>

      {!isFileUploaded ? (
        <FileUpload
          onUploadSuccess={() => {
            setIsFileUploaded(true);
            setMessages([{ text: uiText.fileProcessed, sender: "bot" }]);
          }}
        />
      ) : (
        <Chat
          messages={messages}
          isLoading={isLoading}
          isSuggesting={isSuggesting}
          suggestions={suggestions}
          submitQuery={submitQuery}
          setMessages={setMessages}
          uiText={uiText}
        />
      )}
    </div>
  );
}

export default ChatPage;
