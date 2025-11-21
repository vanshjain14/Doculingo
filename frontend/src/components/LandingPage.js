// src/components/LandingPage.js

import React from "react";
import { Link } from "react-router-dom"; // Used for navigation

function LandingPage() {
  return (
    <div className="landing-page">
      <div className="landing-content">
        <h1>Welcome to DocuLingo AI</h1>
        <p className="subtitle">
          Your Private, Offline, Multilingual Document Assistant
        </p>
        <Link to="/chat" className="start-button">
          Get Started
        </Link>
      </div>
    </div>
  );
}

export default LandingPage;
