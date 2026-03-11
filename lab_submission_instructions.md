# Lab Submission Instruction

## BBT 4206 - CAT 1 (Takeaway) - Due Date: 11th March 2026

## Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:** lab-on-serving-ml-models-202509-b

**Member 1:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        | 149008      |
| **Name**                                                                                                                              | Godwill Kiplagat |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | Implemented the entire lab on the [main branch](https://github.com/BI-course/lab-on-serving-ml-models-202509-b/tree/main):<br/>**Part A:** Updated `api.py` with RESTful API endpoints for all 4 classifiers (Naive Bayes, KNN, SVM, Random Forest) plus the existing Decision Tree classifier and regressor. Each endpoint handles JSON input, applies the correct preprocessing (label encoding, one-hot encoding, scaling), and returns predictions.<br/>**Part B:** Added a product recommender endpoint based on Apriori association rules from the Groceries dataset. The endpoint accepts a list of products and returns recommended products with confidence and lift scores.<br/>**Frontend:** Created HTML test pages with forms for each API endpoint (NB, KNN, SVM, RF classifiers and Recommender). Updated `index.html` to list all 7 endpoints.<br/>**Public Hosting:** Updated the Streamlit app (`streamlit-sharing-using-streamlit/app.py`) with a multi-tab dashboard serving all models and the recommender. Deployed to Streamlit Community Cloud. >**What I learned:** How to serve ML models through RESTful APIs using Flask, how to handle preprocessing pipelines (label encoders, scalers, one-hot encoders) at inference time, the importance of case-sensitive file paths when deploying to Linux-based cloud platforms, and how to containerize Flask apps with Gunicorn + Nginx for production. |

**Member 2:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        |  093624           |
| **Name**                                                                                                                              | David Wahome Waweru            |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | Implemented and tested the decision tree classifier and regressor endpoints, prepared sample JSON payloads, created individual test HTML pages for DT models, assisted with association rules CSV loading and recommender logic.I also updated api.py with endpoints for all four required classifiers (Naive Bayes, kNN, SVM, Random Forest), implemented the association rules recommender, added the advanced k-Means cluster predictor endpoint, added comprehensive error handling (400/422 responses for missing/invalid input), created a unified dashboard in frontend/index.html with vanilla JavaScript fetch calls demonstrating multiple endpoints, and prepared the local Docker + Gunicorn setup.

   **Link:**    https://github.com/BI-course/lab-on-serving-ml-models-202509-b.git

   **What I learned:**    How critical proper input validation and meaningful error responses are for real-world API usability, why Flask endpoint name collisions happen and how to fix them using unique function names or explicit endpoint parameters, the importance of CORS configuration for browser-based frontends, the API delivery layer often matters more to stakeholders than the model accuracy itself and how to structure a simple vanilla JS dashboard to consume the API

**Member 3:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        |             |
| **Name**                                                                                                                              |             |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |             |

**Member 4:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        |             |
| **Name**                                                                                                                              |             |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |             |

**Member 5:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        |             |
| **Name**                                                                                                                              |             |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |             |

## Chosen Level of Difficulty

**Specify the chosen level of difficulty** (baseline, intermediate, or advanced): Advanced

## Video Demonstration

Submit the link to a short video (not more than 5 minutes) demonstrating your solution. Please ensure that the lecturer has rights to view the video.

Note that you are required to submit the link to the video and NOT the video itself. The video should NOT be uploaded to your repository—that would be a misuse of GitHub.

**Link to the video:**
https://drive.google.com/drive/folders/1de3ZC7eaHRYE6SEfH5TcuPII4_x6uIFc?usp=drive_link

## Public URL

**Specify the public URL to the Gradio/Streamlit App**: https://serving-ml-models-group-b.streamlit.app/

---

## Scenario

You have trained various models for regression and classification problems.
You have also used k-Means clustering to identify clusters of clients.
Lastly, you have created association rules to identify products that are frequently bought together.

You are now required to create an API that serves the models and gives recommendations based on the association rules.
This API can be used by systems developed by other teams in the organization using different programming languages.

### Part A

- Refer to the lab on Regression and Classification available [here](https://github.com/course-files/RegressionAndClassification) to understand how the following models were trained:
  - Decision tree regressor
  - Decision tree classifier
  - Naive Bayes classifier
  - K-Nearest Neighbors (kNN) classifier
  - Support Vector Machine (SVM) classifier
  - Random Forest classifier
- Update [api.py](api.py) to include end-points to serve the following models loaded from disk:
  - Naive Bayes classifier
  - K-Nearest Neighbors (kNN) classifier
  - Support Vector Machine (SVM) classifier
  - Random Forest classifier

### Part B

- Refer to the lab on Clustering and Association Rule Mining available [here](https://github.com/course-files/ClusteringandAssociationRuleMining) to understand how the following clusters and association rules were created:
  - k-Means clustering
  - Apriori algorithm for association rule mining based on the "groceries" dataset by Hahsler et al. (2011) that contains 9,835 market basket transactions
- Update [api.py](api.py) to include end-points to serve the following:
  - A recommender that recommends products to a client based on the association rules created in the previous lab. The association rules should be loaded from disk.
  - A classifier that predicts the cluster to which a client belongs to. The classifier should be loaded from disk.

**Note 1:** **`api.py` is NOT production-grade as it is.** It is only meant for demonstration purposes.
Scalability and security must be taken into consideration before deploying an API in a production environment.

**Note 2:** Students often treat the API as an afterthought, focusing only on training ML models.
In practice, the API is the product — it is how others interact with your model.
The "hidden" learning here is that the delivery mechanism (API design, usability, error handling, and even documentation) often matters more to stakeholders in the industry than the models themselves.

**Baseline (Required):**

- Update [api.py](api.py) to include end-points to serve at least three of the models trained in the previous labs since Business Intelligence I.

**Intermediate (Recommended):**

- Update [api.py](api.py) to include end-points to serve the following models:
  - Naive Bayes classifier
  - K-Nearest Neighbors (kNN) classifier
  - Support Vector Machine (SVM) classifier
  - Random Forest classifier

- Update [api.py](api.py) to include end-points to serve the following:
  - A recommender that recommends products to a client based on the association rules created in the previous lab

**Advanced (Optional):**

- Update [api.py](api.py) to include end-points to serve the following:
  - A classifier that predicts the cluster to which a client belongs to
- Create a web page(s) (Basic HTML, CSS, and Vanilla JavaScript) that demonstrates the use of the API
- Implement basic error handling (e.g., missing inputs).
- Flask comes with a development server (good for testing in development, bust unsafe for production).
In production, you need a server that can handle many users, concurrency, and failures. That is where **Gunicorn** comes in. It is a Web Server Gateway Interface (WSGI) server built for production.
- Instead of installing Flask + Gunicorn manually on your machine, you put everything into a Docker image. That image is like a sealed box: it contains your Python code, dependencies, and Gunicorn.
When you run the container, it behaves like a lightweight server.
- How the pieces should fit together:
  - Create a Dockerfile to tell Docker how to build the image (Python, install dependencies, run Gunicorn).
  - Build the Docker image: A one-time process to create a reusable package.
  - Run the container: Starts Gunicorn, serving your Flask app (runs the Flask app and handles requests).

- Dockerize (wrap everything into a container that can run anywhere) your Flask API using Gunicorn to make it "production-ready". Make use of a reverse-proxy (Nginx) to shield Gunicorn from the Internet.

- Why this is production-friendly
  - Portability: the same image runs anywhere (laptop, server, cloud).
  - Consistency: no “but it works on my machine” problems.
  - Isolation: your app runs in its own environment, safe from system changes.
  - Scalability: you can run more containers if you need more capacity.

**For all 3 levels (baseline, intermediate, and advanced):**

- Host any one of the models publicly as either a Gradio app on Hugging Face Spaces or a Streamlit App on Streamlit Community Cloud.

### Grading Approach

The lab will be marked out of 10 marks.

- Exceeding Expectations: 9–10 marks
- Meeting Expectations: 6–8 marks
- Approaching Expectations: 3–5 marks
- Below Expectations: 0–2 marks

The marks also depend on the chosen level of difficulty such that:

- Advanced (Distinction) >= 8.6 marks
- Intermediate (Merit) 7.5–8.5 marks
- Baseline (Pass) >= 6 marks
