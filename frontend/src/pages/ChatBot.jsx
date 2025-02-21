import { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const ChatBot = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I assist you today?" },
  ]);
  const [email, setEmail] = useState("");
  const [session_id, setSession_id] = useState("");
  const [sessions, setSessions] = useState([]); // Will now store { sessionId, firstMessage }
  const [selectedSession, setSelectedSession] = useState(null);
  const [question, setQuestion] = useState("");

  useEffect(() => {
    setEmail(JSON.parse(localStorage.getItem("email")));
    setSession_id(localStorage.getItem("sessionId"));
  }, []);

  useEffect(() => {
    if (email) {
      axios
        .get(`http://localhost:5002/get_chat_sessions`, { params: { email } })
        .then(async (response) => {
          if (response.data.sessions) {
            // Fetch first message for each session
            const sessionPromises = response.data.sessions.map(
              async (sessionId) => {
                const chatResponse = await axios.get(
                  `http://localhost:5002/get_chat_history`,
                  {
                    params: { email, session_id: sessionId },
                  }
                );

                const firstMessage =
                  chatResponse.data.chat_history?.[0]?.message ||
                  `Session ${sessionId}`;
                return { sessionId, firstMessage };
              }
            );

            const updatedSessions = await Promise.all(sessionPromises);
            setSessions(updatedSessions);
          }
        })
        .catch((error) => console.error("Error fetching sessions:", error));
    }
  }, [email]);

  const fetchChatHistory = (sessionId) => {
    axios
      .get(`http://localhost:5002/get_chat_history`, {
        params: { email, session_id: sessionId },
      })
      .then((response) => {
        if (
          response.data.chat_history &&
          response.data.chat_history.length > 0
        ) {
          const formattedMessages = response.data.chat_history
            .map((chat) => [
              { sender: "user", text: chat.message },
              { sender: "bot", text: chat.response },
            ])
            .flat();

          // Set the session ID based on the first message
          const firstMessage = response.data.chat_history[0].message;
          setSelectedSession(firstMessage);
          localStorage.setItem("sessionId", firstMessage);

          setMessages([
            { sender: "bot", text: `Session "${firstMessage}" loaded.` },
            ...formattedMessages,
          ]);
        } else {
          setMessages([
            {
              sender: "bot",
              text: `Session ${sessionId} loaded, but no messages found.`,
            },
          ]);
          setSelectedSession(sessionId);
        }
      })
      .catch((error) => console.error("Error fetching chat history:", error));
  };

  const sendMessage = async () => {
    if (!question.trim()) return;

    const userMessage = { sender: "user", text: question };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");

    try {
      const response = await axios.post("http://localhost:5002/ask", {
        email,
        question,
        session_id: selectedSession, // Uses the updated session ID
      });

      if (response.data.session_id) {
        localStorage.setItem(
          "sessionId",
          JSON.stringify(response.data.session_id)
        );
      }

      const botReply = {
        sender: "bot",
        text: response.data.answer || "Please reframe your question",
      };
      setMessages((prev) => [...prev, botReply]);
    } catch (error) {
      
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error: Unable to fetch response" },
      ]);
    }
  };


  const handleLogOut = () => {
    localStorage.clear();
    navigate("/login");
  }

  return (
    <>
     <button
        onClick={handleLogOut}
        className="absolute top-4 right-4 bg-red-500 text-white p-2 rounded-lg hover:bg-red-600 z-50"
      >
        Logout
      </button>
    
      <div className="flex flex-wrap justify-center">
        {/* Chat Sessions */}
        <div className="hidden md:block relative w-[30%] md:w-[30%] ml-5">
          <div className="relative z-1 p-0.5 rounded-2xl bg-conic-gradient">
            <div className="relative bg-n-8 rounded-[1rem] min-h-[610px]">
              <h2 className="text-lg font-semibold text-center mb-3 pt-2">Previous Charts</h2>
              <ul>
                {sessions.map((session) => (
                  <li
                    key={session.sessionId}
                    className={`cursor-pointer p-2 m-1.5  rounded-lg ${
                      selectedSession === session.firstMessage
                        ? "bg-gray-700"
                        : "bg-gray-800"
                    } hover:bg-gray-600`}
                    onClick={() => fetchChatHistory(session.sessionId)}
                  >
                    {session.firstMessage}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Chat Box */}
        <div className="relative w-[70%] mx-auto md:w-[65%] mr-1">
          <div className="relative z-1 p-0.5 rounded-2xl bg-conic-gradient">
            <div className="relative bg-n-8 rounded-[1rem] min-h-[600px]">
              <div className="flex flex-col overflow-y-auto py-6 px-3 space-y-2 h-[550px]">
                {messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`flex p-2 rounded-lg max-w-[75%] ${
                      msg.sender === "user"
                        ? "bg-blue-500 text-white self-end"
                        : "bg-gray-200 text-black self-start"
                    }`}
                  >
                    {msg.text}
                  </div>
                ))}
              </div>
              <div className="flex p-2 border-t border-gray-300">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Type a Message..."
                  className="flex flex-1 p-2 border border-gray-300 rounded-lg focus:outline-none"
                  onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                />
                <button
                  className="ml-2 bg-blue-500 text-white p-2 rounded-lg hover:bg-blue-600"
                  onClick={sendMessage}
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default ChatBot;
