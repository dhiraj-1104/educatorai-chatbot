import ButtonSvg from "../assets/svg/ButtonSvg";
import { Link } from "react-router-dom";

const Button = ({ className, href, onClick, children, px, white }) => {
  const classes = `button relative inline-flex items-center justify-center h-11 transition-color hover:text-color-1 ${
    px || "px-7"
  } ${white ? "text-n-8" : "text-n-1"} ${className || ""}`;

  const spanClasses = "relative z-10";

  const renderButton = () => (
    
      <button className={classes} onClick={onClick} type="button">
        <span className={spanClasses}>{children}</span>
        {ButtonSvg(white)}
      </button>
   
  );

  const renderLink = () => (
    <Link to={href}>
      <span className={spanClasses}>{children}</span>
    </Link>
  );

  return href ? renderLink() : renderButton();
};

export default Button;
