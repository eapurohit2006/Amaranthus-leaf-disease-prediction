import './Help.css'

function Help() {
  return (
    <div className="help-page">
      <div className="help-container">
        {/* Header Section */}
        <div className="help-header-card">
          <div className="header-content">
            <div className="header-icon">❓</div>
            <div className="header-text">
              <h1>Help & Support</h1>
              <p>Everything you need to know to get the most out of LeafScan</p>
            </div>
          </div>
        </div>

        {/* How to Use Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">📖</span>
            <h2>How to Use the System</h2>
          </div>
          <p className="section-subtitle">Step-by-step guide to detecting plant diseases</p>
          <div className="steps-list">
            <div className="step-item">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Create an Account</h3>
                <p>Sign up for a free LeafScan account to start using the disease detection features</p>
              </div>
            </div>
            <div className="step-item">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Capture or Upload Image</h3>
                <p>Take a clear photo of the amaranthus leaf or upload an existing image from your device</p>
              </div>
            </div>
            <div className="step-item">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Analyze the Image</h3>
                <p>Click the "Analyze Image" button to process your leaf photo through our AI model</p>
              </div>
            </div>
            <div className="step-item">
              <div className="step-number">4</div>
              <div className="step-content">
                <h3>View Results</h3>
                <p>Review the disease detection results, confidence level, and treatment recommendations</p>
              </div>
            </div>
            <div className="step-item">
              <div className="step-number">5</div>
              <div className="step-content">
                <h3>Take Action</h3>
                <p>Follow the treatment recommendations to address any detected diseases</p>
              </div>
            </div>
            <div className="step-item">
              <div className="step-number">6</div>
              <div className="step-content">
                <h3>Track History</h3>
                <p>Monitor your prediction history to track plant health over time</p>
              </div>
            </div>
          </div>
        </div>

        {/* Best Practices Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">📸</span>
            <h2>Best Practices for Image Upload</h2>
          </div>
          <p className="section-subtitle">Tips to get the most accurate results</p>
          <div className="tips-grid">
            <div className="tip-card">
              <div className="tip-icon">☀️</div>
              <h3>Use Natural Light</h3>
              <p>Take photos in bright, natural daylight. Avoid harsh shadows and direct sunlight that can wash out details</p>
            </div>
            <div className="tip-card">
              <div className="tip-icon">🎯</div>
              <h3>Focus on the Leaf</h3>
              <p>Ensure the leaf is in sharp focus. Blurry images reduce detection accuracy</p>
            </div>
            <div className="tip-card">
              <div className="tip-icon">🖼️</div>
              <h3>Simple Background</h3>
              <p>Use a plain background (white paper or soil) to help the AI focus on the leaf</p>
            </div>
            <div className="tip-card">
              <div className="tip-icon">📏</div>
              <h3>Fill the Frame</h3>
              <p>Make the leaf fill most of the image frame. Avoid multiple leaves in one photo</p>
            </div>
            <div className="tip-card">
              <div className="tip-icon">📱</div>
              <h3>Hold Steady</h3>
              <p>Keep your camera or phone steady to avoid motion blur</p>
            </div>
            <div className="tip-card">
              <div className="tip-icon">🔄</div>
              <h3>Multiple Angles</h3>
              <p>If unsure, take photos from different angles for better analysis</p>
            </div>
          </div>
        </div>

        {/* Farmer Tip Box */}
        <div className="farmer-tip-box">
          <div className="tip-box-icon">🌾</div>
          <div className="tip-box-content">
            <h3>Pro Tip for Farmers</h3>
            <p>Clear, bright leaf images give the best results! Take photos in the morning when natural light is optimal.</p>
          </div>
        </div>

        {/* FAQ Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">❓</span>
            <h2>Common Issues & Troubleshooting</h2>
          </div>
          <p className="section-subtitle">Frequently asked questions and solutions</p>
          <div className="faq-list">
            <div className="faq-item">
              <div className="faq-question">
                <span className="faq-icon">🔍</span>
                <h3>Why is my image not uploading?</h3>
              </div>
              <div className="faq-answer">
                <p>Make sure your image is in JPG, PNG, or JPEG format and is under 10MB in size. Check your internet connection and try again.</p>
              </div>
            </div>
            <div className="faq-item">
              <div className="faq-question">
                <span className="faq-icon">⚠️</span>
                <h3>I see an "unclear image" warning</h3>
              </div>
              <div className="faq-answer">
                <p>This usually means the image is blurry or the leaf isn't clearly visible. Try taking a new photo with better lighting and focus.</p>
              </div>
            </div>
            <div className="faq-item">
              <div className="faq-question">
                <span className="faq-icon">🎯</span>
                <h3>How accurate are the predictions?</h3>
              </div>
              <div className="faq-answer">
                <p>Our AI model has been trained on thousands of images and typically achieves 90%+ accuracy. However, results may vary based on image quality.</p>
              </div>
            </div>
            <div className="faq-item">
              <div className="faq-question">
                <span className="faq-icon">🔒</span>
                <h3>Is my data secure?</h3>
              </div>
              <div className="faq-answer">
                <p>Yes, we take data privacy seriously. Your images and account information are securely stored and never shared with third parties.</p>
              </div>
            </div>
            <div className="faq-item">
              <div className="faq-question">
                <span className="faq-icon">📱</span>
                <h3>Can I use this on my phone?</h3>
              </div>
              <div className="faq-answer">
                <p>Yes! LeafScan works on all devices including smartphones, tablets, and computers. Just open it in your web browser.</p>
              </div>
            </div>
            <div className="faq-item">
              <div className="faq-question">
                <span className="faq-icon">💊</span>
                <h3>Should I follow all treatment recommendations?</h3>
              </div>
              <div className="faq-answer">
                <p>For critical cases, we recommend consulting with a professional agronomist or plant pathologist in addition to following our recommendations.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Support Contact Section */}
        <div className="content-section">
          <div className="section-header">
            <span className="section-icon">📧</span>
            <h2>Support & Contact</h2>
          </div>
          <p className="section-subtitle">Get in touch with our support team</p>
          <div className="support-cards">
            <div className="support-card">
              <div className="support-icon">📧</div>
              <h3>Email Support</h3>
              <p>support@leafscan.com</p>
              <p className="support-note">We typically respond within 24 hours</p>
            </div>
            <div className="support-card">
              <div className="support-icon">💬</div>
              <h3>WhatsApp</h3>
              <p>+1 (555) 123-4567</p>
              <p className="support-note">Available Monday-Friday, 9 AM - 5 PM</p>
            </div>
            <div className="support-card">
              <div className="support-icon">🌐</div>
              <h3>Documentation</h3>
              <p>Visit our help center</p>
              <p className="support-note">Find detailed guides and tutorials</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Help
