<h1>Predicting Instagram Post Engagement Using Machine Learning</h1>

<h2>Project Overview</h2>
<p>This project leverages machine learning techniques to predict <strong>Instagram post engagement</strong> based on historical data, aiming to help businesses and content creators optimize their content strategy for better user interaction. By analyzing various post features such as media type, caption, and engagement metrics (likes, comments, shares), the model predicts the <strong>engagement rate</strong> for Instagram posts. The model is trained on a dataset sourced from Instagram’s Graph API and evaluated using multiple regression models.</p>

<h3>Objectives:</h3>
<ul>
    <li>Predict the engagement of Instagram posts based on historical data.</li>
    <li>Help businesses optimize their content strategy by understanding which types of posts perform well.</li>
    <li>Evaluate multiple machine learning models to determine the best approach for predicting engagement.</li>
</ul>

<h2>Dataset</h2>
<p>The dataset used in this project is obtained through <strong>Instagram’s Graph API</strong>. It contains data about posts from Instagram over the past two years, including:</p>
<ul>
    <li><strong>Post ID</strong>: Unique identifier for each post.</li>
    <li><strong>Media Type</strong>: Type of content (photo, video, carousel).</li>
    <li><strong>Caption</strong>: The text that accompanies the post.</li>
    <li><strong>Timestamp</strong>: Time and date when the post was published.</li>
    <li><strong>Engagement Metrics</strong>: Likes, comments, shares, saves, and reach.</li>
    <li><strong>Additional Features</strong>: Hashtag count, caption length, and time-based features (hour, day of the week, etc.).</li>
</ul>

<p>The dataset is preprocessed by handling missing values, normalizing continuous variables, and generating additional features (e.g., time of posting and sentiment analysis of captions).</p>

<h2>Models Used</h2>
<p>Three machine learning models are implemented and evaluated:</p>
<ul>
    <li><strong>Linear Regression</strong>: Used as a baseline model for predicting post engagement based on input features.</li>
    <li><strong>Random Forest</strong>: An ensemble method that can capture non-linear relationships between features.</li>
    <li><strong>XGBoost</strong>: A gradient-boosting algorithm known for its high performance in regression tasks.</li>
</ul>

<h3>Evaluation:</h3>
<p>Each model is evaluated using the following metrics:</p>
<ul>
    <li><strong>Mean Absolute Error (MAE)</strong>: Measures the average magnitude of errors between predicted and actual engagement.</li>
    <li><strong>Mean Squared Error (MSE)</strong>: Penalizes larger errors by squaring them, providing insight into model accuracy.</li>
    <li><strong>R-squared (R²)</strong>: Indicates the proportion of variance in engagement explained by the model. Higher R² values indicate better model performance.</li>
</ul>

<h2>Project Workflow</h2>
<p>1. <strong>Data Collection:</strong> The Instagram Graph API is used to collect post data, including engagement metrics and post details. The data is then stored locally and preprocessed for training the machine learning models.</p>

<p>2. <strong>Data Preprocessing:</strong> Missing values are handled by filling with the median for each respective feature. Numerical features such as likes and shares are normalized to scale them between 0 and 1. New features are created, including time-based features (e.g., hour of posting, day of the week), and sentiment analysis of captions.</p>

<p>3. <strong>Model Training:</strong> Multiple models (Linear Regression, Random Forest, and XGBoost) are trained on the dataset. Hyperparameter tuning is performed using GridSearchCV for Random Forest and XGBoost to optimize model performance.</p>

<p>4. <strong>Model Evaluation:</strong> The models are evaluated using 5-fold cross-validation to assess their performance and avoid overfitting. Evaluation metrics such as MAE, MSE, and R² are used to compare model performance.</p>

<p>5. <strong>Results:</strong> The results showed that XGBoost outperformed both Random Forest and Linear Regression in predicting post engagement with the highest R² (0.88) and the lowest MAE (0.0031).</p>

<h2>How to Run the Project</h2>

<h3>1. Clone the Repository:</h3>
<p>Clone this repository to your local machine using the following command:</p>
<pre>git clone https://github.com/your-username/instagram-engagement-prediction.git</pre>

<h3>2. Install Dependencies:</h3>
<p>This project requires Python 3.7 or higher. Install the required Python libraries using pip:</p>
<pre>pip install -r requirements.txt</pre>
<p>The <code>requirements.txt</code> includes dependencies like <code>pandas</code>, <code>scikit-learn</code>, <code>xgboost</code>, <code>lightgbm</code>, and others.</p>

<h3>3. Run the Code:</h3>
<p>
    1. <strong>Data Collection:</strong> To collect data from the Instagram Graph API, ensure you have the necessary credentials and API access.<br>
    2. <strong>Data Preprocessing:</strong> Run the preprocessing script to clean the data, normalize the features, and create additional features like caption sentiment and time-based features.<br>
    3. <strong>Model Training:</strong> Use the model training script to train the models (Linear Regression, Random Forest, and XGBoost) on the preprocessed data. The script will output model performance metrics, including MAE, MSE, and R².<br>
    4. <strong>Results:</strong> After training, the results will be displayed in the console and saved as graphical reports (feature importance, predictions vs. actual values, model evaluation).
</p>

<h2>Tools Used</h2>
<ul>
    <li><strong>Programming Language:</strong> Python 3.7+</li>
    <li><strong>Libraries:</strong> <code>scikit-learn</code>, <code>xgboost</code>, <code>lightgbm</code>, <code>pandas</code>, <code>matplotlib</code>, <code>seaborn</code></li>
    <li><strong>Data Source:</strong> Instagram Graph API</li>
</ul>

<h2>Conclusion</h2>
<p>This project provides a comprehensive approach to predicting Instagram post engagement using machine learning. By analyzing historical data, we built models that can predict post performance, which is valuable for businesses and content creators looking to optimize their content strategies. The results show that <strong>XGBoost</strong> is the best model for predicting engagement, outperforming other models in terms of accuracy.</p>

<p>This project demonstrates the practical application of machine learning to real-world problems and provides insights that can drive better content decisions for social media marketing.</p>

<h2>Future Work</h2>
<ul>
    <li><strong>Real-Time Prediction:</strong> Implementing real-time prediction models for businesses to dynamically adjust their posting strategies.</li>
    <li><strong>Deep Learning:</strong> Exploring deep learning models for more complex, non-linear relationships.</li>
    <li><strong>Other Social Media Platforms:</strong> Extending this analysis to other platforms such as Twitter and YouTube.</li>
</ul>

<h2>Additional Resources</h2>
<p>For a more in-depth explanation of the project, you can refer to the following resources:</p>
<ul>
    <li><strong>PowerPoint Presentation:</strong> <a href="https://coyotesusd-my.sharepoint.com/:p:/g/personal/alex_l_jensen_coyotes_usd_edu/ER_b-rD82OZGp2Sy7sN_n30BXJFJvu_haeVlKnefaPODdQ?e=2qKP4E" target="_blank">Click here to view the PowerPoint presentation</a></li>
    <li><strong>Project Paper (PDF):</strong> <a href="https://drive.google.com/file/d/1NVAzZ3pMKwe2W0Ab9yanAVpQFlM-fAwt/view?usp=sharing" target="_blank">Click here to view the project paper</a></li>
</ul>
