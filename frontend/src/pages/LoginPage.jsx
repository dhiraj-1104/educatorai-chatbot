import Button from "../component/Button";
import { Link, useNavigate } from "react-router-dom";
import useAlert from "../hooks/useAlert";
import Alert from "../component/Alert";
import axios from "axios";
import { useState } from "react";

const LoginPage = () => {
  const { alert, showAlert, hideAlert } = useAlert();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  //Generate random sessionid
  const generateSessionId = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  };
  

  // Function to handle user login
  const handleLogin = async () => {
    if (email == "" && password == "") {
      showAlert({
        show: true,
        text: "Email and Password Required",
        type: "danger",
      });
      setTimeout(() => {
        hideAlert();
      }, 3000);
      return;
    }

    try {
      const response = await axios.post("http://localhost:5002/login", {
        email,
        password,
      });
      showAlert({
        show: true,
        text: response.data.message || "Login",
        type: "success",
      });

      localStorage.setItem("sessionId",generateSessionId());
      localStorage.setItem("email",JSON.stringify(email));
      

      setTimeout(() => {
        navigate("/chatbot");
        hideAlert();
      }, 1000);
    } catch (error) {
      showAlert({
        show: true,
        text: error.response?.data?.error || "Failed",
        type: "danger",
      });
      setTimeout(() => {
        hideAlert();
      }, 3000);
    }
  };

  return (
    <>
      <div>{alert.show && <Alert text={alert.text} type={alert.type} />}</div>
      <div className="relative  max-w-[23rem]  mx-auto md:max-w-lg xl:mb-24 mt-24 md:mt-10">
        <div className=" relative z-1 p-0.5 rounded-2xl bg-conic-gradient">
          <div className="relative bg-n-8 rounded-[1rem]">
            <form onSubmit={(e) => e.preventDefault()} className="px-10 py-5">
              <h1 className="h1 text-center">Login</h1>
              <label className="">Email</label>
              <input
                type="email"
                className="border border-n-1 rounded-lg mx-auto w-full mb-5 py-2 pl-2"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <label>Password</label>
              <input
                type="password"
                className="border border-n-1 rounded-lg mx-auto  w-full mb-2 py-2 pl-2"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <p className=" text-right mb-4 text-sm right-0">
                Forgot Password?
              </p>
              <Button white className=" w-full mb-2" onClick={handleLogin}>
                Login
              </Button>
              <p className="mb-4 body-1 text-center">
                Don't have account.
                <Link
                  to="/register"
                  className="text-blue-500 hover:text-blue-700 "
                >
                  Create account!
                </Link>
              </p>
            </form>
          </div>
        </div>
      </div>
    </>
  );
};

export default LoginPage;
