import Button from "../component/Button";
import { Link, useNavigate } from "react-router-dom";
import useAlert from "../hooks/useAlert";
import Alert from "../component/Alert";
import { useState } from "react";
import axios from "axios";

const RegisterPage = () => {
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const { alert, showAlert, hideAlert } = useAlert();
  const navigate = useNavigate();

  // Function to handle user signup
  const handleSignup = async () => {
    console.log("Signup button clicked!");
    if (password !== confirmPassword) {
      showAlert({
        show: true,
        text: "Passwords do not match",
        type: "danger",
      });
      setTimeout(() => {
        hideAlert();
      }, 3000);
      return;
    }

    if(email == "" && password == "" && username == ""){
      showAlert({
        show:true,
        text : "All the fields are Required",
        type: "danger"
      })
      setTimeout(() => {
        hideAlert();
      }, 3000);
      return;
    }

    try {
      const response = await axios.post(
        "http://localhost:5002/create_account",
        {
          email,
          username,
          password,
        }
      );
      showAlert({
        show: true,
        text: response.data.message || "Account created successfully!",
        type: "success",
      });

      setTimeout(() => {
        hideAlert();
        navigate("/login");
      }, 3000);
    } catch (error) {
      showAlert({
        show: true,
        text: error.response?.data?.error || "Sign up failed" ,
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
      <div className="relative  max-w-[23rem]  mx-auto md:max-w-lg xl:mb-24 mt-10">
        <div className=" relative z-1 p-0.5 rounded-2xl bg-conic-gradient">
          <div className="relative bg-n-8 rounded-[1rem]">
            <form onSubmit={(e) => e.preventDefault()} className="px-10 py-5">
              <h1 className="h2 mb-6 text-center">Create Account</h1>
              <label className="">Username</label>
              <input
                type="text"
                className="border border-n-1 rounded-lg mx-auto w-full mb-3 py-2 pl-2"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
              <label className="">Email</label>
              <input
                type="email"
                className="border border-n-1 rounded-lg mx-auto w-full mb-3 py-2 pl-2"
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
              <label>Confirm Password</label>
              <input
                type="password"
                className="border border-n-1 rounded-lg mx-auto  w-full mb-6 py-2 pl-2"
                placeholder="Confirm Password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />

              <Button white className=" w-full mb-2" onClick={handleSignup}>
                Create Account
              </Button>
              <p className="mb-4 body-1 text-center">
                Already have an Account.
                <Link
                  to="/login"
                  className="text-blue-500 hover:text-blue-700 "
                >
                  Login!
                </Link>
              </p>
            </form>
          </div>
        </div>
      </div>
    </>
  );
};

export default RegisterPage;
