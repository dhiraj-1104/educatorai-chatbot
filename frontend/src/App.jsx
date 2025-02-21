import Header from "./component/Header";
import Hero from "./component/Hero";
import { Routes, Route } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import ChatBot from "./pages/ChatBot";

const App = () => {
  return (
    <>
      <div className="pt-[4.75rem] lg:pt-[5.25rem] overflow-hidden">
        <Header />

        <Routes>
          <Route path="/" element={<Hero />} />

          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          <Route path="/chatbot" element={<ChatBot />} />
        </Routes>
      </div>
    </>
  );
};

export default App;
