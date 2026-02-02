import './About.css'

function About() {
  return (
    <div className="about-page">
      <div className="about-container">
        {/* Header Section */}
        <div className="about-header-card">
          <div className="header-content">
            <div className="header-icon">🍃</div>
            <div className="header-text">
              <h1>About LeafScan</h1>
              <p>Empowering farmers with AI-powered plant disease detection for healthier crops and better yields</p>
            </div>
          </div>
        </div>

        {/* Mission & Purpose Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">🎯</span>
            <h2>Mission & Purpose</h2>
          </div>
          <p className="section-subtitle">Our commitment to transforming agriculture through technology</p>
          <div className="section-content">
            <p>
              LeafScan is dedicated to helping farmers and agricultural enthusiasts quickly identify and treat 
              plant diseases using cutting-edge artificial intelligence. Our mission is to make advanced disease 
              detection accessible to everyone, ensuring healthier crops and sustainable farming practices.
            </p>
            <p>
              We believe that early detection is key to preventing crop loss and reducing the need for excessive 
              pesticide use. By leveraging AI technology, we provide instant, accurate diagnoses that enable 
              timely intervention and better crop management.
            </p>
          </div>
        </div>

        {/* Why AI Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">🤖</span>
            <h2>Why AI for Leaf Disease Detection?</h2>
          </div>
          <p className="section-subtitle">The power of machine learning in agriculture</p>
          <div className="section-content">
            <p>
              Traditional disease detection methods can be time-consuming and require expert knowledge. Our AI-powered 
              system offers:
            </p>
            <ul className="feature-list">
              <li>
                <span className="list-icon">⚡</span>
                <div>
                  <strong>Instant Results:</strong> Get disease diagnosis in seconds, not days
                </div>
              </li>
              <li>
                <span className="list-icon">🎯</span>
                <div>
                  <strong>High Accuracy:</strong> Advanced deep learning models trained on thousands of leaf images
                </div>
              </li>
              <li>
                <span className="list-icon">🌱</span>
                <div>
                  <strong>Accessibility:</strong> No need for expensive laboratory tests or expert consultation
                </div>
              </li>
              <li>
                <span className="list-icon">📱</span>
                <div>
                  <strong>Convenience:</strong> Available 24/7 from any device with internet connection
                </div>
              </li>
              <li>
                <span className="list-icon">💰</span>
                <div>
                  <strong>Cost-Effective:</strong> Reduce crop loss and unnecessary pesticide expenses
                </div>
              </li>
            </ul>
          </div>
        </div>

        {/* Technologies Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">💻</span>
            <h2>Technologies Used</h2>
          </div>
          <p className="section-subtitle">Built with modern, reliable technologies</p>
          <div className="tech-grid">
            <div className="tech-card">
              <div className="tech-icon">⚛️</div>
              <h3>React</h3>
              <p>Modern frontend framework for responsive user interface</p>
            </div>
            <div className="tech-card">
              <div className="tech-icon">🐍</div>
              <h3>Python FastAPI</h3>
              <p>High-performance backend API for fast predictions</p>
            </div>
            <div className="tech-card">
              <div className="tech-icon">🧠</div>
              <h3>TensorFlow/Keras</h3>
              <p>Deep learning framework powering our AI model</p>
            </div>
            <div className="tech-card">
              <div className="tech-icon">🔬</div>
              <h3>ResNet50V2</h3>
              <p>State-of-the-art convolutional neural network for image classification</p>
            </div>
          </div>
        </div>

        {/* About Creator Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">👨‍💻</span>
            <h2>About the Creator</h2>
          </div>
          <p className="section-subtitle">Passionate about AI and agriculture</p>
          <div className="creator-card">
            <div className="creator-info">
              <div className="creator-icon">🌾</div>
              <div>
                <h3>Rohit</h3>
                <p className="creator-role">Computer Science Engineering Student</p>
                <p>
                  Rohit is a CSE student passionate about building AI solutions for agriculture. LeafScan represents 
                  his vision of using technology to solve real-world problems faced by farmers. Through this project, 
                  he aims to bridge the gap between cutting-edge AI research and practical agricultural applications, 
                  making advanced disease detection tools accessible to everyone.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Future Goals Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">🚀</span>
            <h2>Future Goals & Roadmap</h2>
          </div>
          <p className="section-subtitle">Our vision for the future of LeafScan</p>
          <div className="roadmap-list">
            <div className="roadmap-item">
              <span className="roadmap-icon">🌿</span>
              <div>
                <h4>Expand Disease Coverage</h4>
                <p>Add support for more plant species and disease types</p>
              </div>
            </div>
            <div className="roadmap-item">
              <span className="roadmap-icon">📊</span>
              <div>
                <h4>Enhanced Analytics</h4>
                <p>Detailed crop health tracking and historical analysis</p>
              </div>
            </div>
            <div className="roadmap-item">
              <span className="roadmap-icon">🌍</span>
              <div>
                <h4>Multi-language Support</h4>
                <p>Make the platform accessible to farmers worldwide</p>
              </div>
            </div>
            <div className="roadmap-item">
              <span className="roadmap-icon">📱</span>
              <div>
                <h4>Mobile App</h4>
                <p>Native mobile applications for iOS and Android</p>
              </div>
            </div>
            <div className="roadmap-item">
              <span className="roadmap-icon">🤝</span>
              <div>
                <h4>Expert Integration</h4>
                <p>Connect farmers directly with agricultural experts</p>
              </div>
            </div>
            <div className="roadmap-item">
              <span className="roadmap-icon">🔔</span>
              <div>
                <h4>Smart Notifications</h4>
                <p>Proactive alerts and seasonal disease predictions</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default About
