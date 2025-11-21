import React from "react";
import "./LanguageSelector.css";

function LanguageSelector({ language, onLanguageChange }) {
  return (
    <div className="language-selector">
      <button
        className={language === "en" ? "active" : ""}
        onClick={() => onLanguageChange("en")}
      >
        🇬🇧 English
      </button>

      <button
        className={language === "es" ? "active" : ""}
        onClick={() => onLanguageChange("es")}
      >
        🇪🇸 Español
      </button>

      <button
        className={language === "de" ? "active" : ""}
        onClick={() => onLanguageChange("de")}
      >
        🇩🇪 Deutsch
      </button>
    </div>
  );
}

export default LanguageSelector;
