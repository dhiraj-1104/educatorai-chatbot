import Section from "../component/Section";
import curve from "../assets/hero/curve.png";
import robot from "../assets/hero/robot.jpg";
import Button from "../component/Button";
import heroBackground from '../assets/hero/hero-background.jpg'
import { useNavigate } from "react-router-dom";


const Hero = () => {

  const navigate = useNavigate();

  const navigateTOLogin=()=>{
          navigate("/login");
  }
 
  return (
    <Section
      className="pt-[12rem] -mt-[5.25rem]"
      crosses
      crossOffset="lg:translate-y-[5.25rem]"
      customPaddings
      id="Hero"
    >
      <div className="container relative">
        <div className="relative z-1 max-w-[62rem] mx-auto text-center mb-[4rem] md:mb-20 lg:mb-[6rem]">
          <h1 className="h1 mb-6">
          Your AI-Powered Guide to Smarter Knowledge. <br />
            <span className="inline-block relative">
             - EducatorAi{" "}
              <img
                src={curve}
                alt="curve"
                className="absolute top-full left-0 w-full xl:-mt-2"
                width={624}
                height={28}
              />
            </span>
          </h1>
          <p className="body-1 max-w-3xl mx-auto mb-6 text-n-2 lg:mb-8">
          Experience Intelligent Conversations and Smarter Learning with EducatorAi – Your Ultimate AI Chat Companion!
          </p>
          <Button white onClick={navigateTOLogin}>Try EducatorAi</Button>
        </div>
        <div className="relative  max-w-[23rem] mx-auto md:max-w-5xl xl:mb-24">
          <div className=" relative z-1 p-0.5 rounded-2xl bg-conic-gradient">
            <div className="relative bg-n-8 rounded-[1rem]">
              <div className="h-[1.4rem] bg-n-10 rounded-t-[0.9rem]" />

              <div className="aspect-[33/40] rounded-b-[0.9rem] overflow-hidden md:aspect-[688/490] lg:aspect-[1024/490]">
                <img src={robot} alt="robot" 
                className="w-full scale-[1.7] translate-y-[8%] md:scale-[1] md:-translate-y-[10%] lg:-translate-y-[23%]"
                />
              </div> 
            </div>
          </div>
          <div className="absolute -top-[54%] left-1/2 w-[234%] -translate-x-1/2 md:-top-[46%] md:w-[138%] lg:-top-[104%]" >
            <img src={heroBackground} alt="background" 
            width={1440} height={1800}/>
          </div>
        </div>
      </div>
    </Section>
  );
};

export default Hero;
