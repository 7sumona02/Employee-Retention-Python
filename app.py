import pandas as pd
import joblib
import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your Groq API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Make sure to set this in your .env file

# Load the trained model for employee retention prediction
model = joblib.load('emp-ret-model.pkl')

# Define a function for the Employee Retention Prediction page
def employee_retention_prediction():
    st.title("Employee Retention Prediction")

    # Input fields for the new employee
    satisfaction_level = st.slider('Satisfaction Level', 0.0, 1.0, 0.5)
    last_evaluation = st.slider('Last Evaluation', 0.0, 1.0, 0.5)
    number_project = st.number_input('Number of Projects', min_value=1, max_value=10, value=3)
    average_monthly_hours = st.number_input('Average Monthly Hours', min_value=50, max_value=300, value=150)
    time_spend_company = st.number_input('Time Spent at Company (years)', min_value=1, max_value=10, value=3)
    work_accident = st.selectbox('Work Accident', [0, 1])
    promotion_last_5years = st.selectbox('Promotion in Last 5 Years', [0, 1])
    sales = st.selectbox('Department (Sales)', list(range(5)))  # Assuming 10 departments
    salary = st.selectbox('Salary Level', ['low', 'medium', 'high'])

    # Map salary to numerical value
    salary_map = {'low': 1, 'medium': 2, 'high': 3}
    salary_value = salary_map[salary]

    if st.button('Predict'):
        # Prepare input data for prediction
        input_data = pd.DataFrame([[satisfaction_level, last_evaluation, number_project, 
                                     average_monthly_hours, time_spend_company, 
                                     work_accident, promotion_last_5years, sales, 
                                     salary_value]], 
                                   columns=['satisfaction_level', 'last_evaluation', 'number_project', 
                                            'average_montly_hours', 'time_spend_company', 
                                            'Work_accident', 'promotion_last_5years', 
                                            'sales', 'salary'])
        
        # Make prediction
        prediction_proba = model.predict_proba(input_data)
        prediction = 1 if prediction_proba[0][1] >= 0.5 else 0  # Fixed threshold of 0.5
        
        # Determine if the employee will stay or leave
        retention_probability = "Will leave" if prediction == 1 else "Will stay"
        
        # Display the result
        st.write(f'Prediction: {retention_probability}')
        st.write(f'Probability of Leaving: {prediction_proba[0][1]:.2f}')

# Define a function for the Work-Life Balance Preferences page
def work_life_balance_metrics():
    st.title("Work-Life Balance Preferences")

    # Input fields for work-life balance preferences
    work_hours = st.selectbox('Preferred Working Hours', ['Day', 'Night'])
    flexible_hours = st.number_input('Preferred Number of Working Hours per Day', min_value=1, max_value=24, value=8)
    work_location = st.selectbox('Preferred Work Arrangement', ['Remote', 'In-Office', 'Hybrid'])

    # Additional preferences
    st.write("Please select your preferred work-life balance options:")
    additional_options = st.multiselect(
        'Select Additional Preferences',
        ['Flexible Start Time', 'Shorter Work Weeks', 'More Breaks', 'Work from Different Locations']
    )

    if st.button('Submit Preferences'):
        # Prepare the data for submission
        preferences = {
            "Working Hours": work_hours,
            "Flexible Hours": flexible_hours,
            "Work Arrangement": work_location,
            "Additional Preferences": additional_options
        }
        
        # Display confirmation of submission
        st.success("Your work-life balance preferences have been submitted!")
        st.write(preferences)

# Define a function for the Recognition and Rewards page
def recognition_and_rewards():
    st.title("Recognition and Rewards")

    # Input fields for recognition and rewards
    employee_name = st.text_input("Employee Name")
    recognition_type = st.selectbox("Type of Recognition", ["Verbal Praise", "Bonus", "Promotion", "Award", "Other"])
    feedback = st.text_area("Feedback or Comments")

    if st.button('Submit Recognition'):
        # Prepare the data for submission
        recognition_data = {
            "Employee Name": employee_name,
            "Type of Recognition": recognition_type,
            "Feedback": feedback
        }
        
        # Display confirmation of submission
        st.success("Recognition and reward details have been submitted!")
        st.write(recognition_data)

# Define a function for the Chatbot page
def chatbot():
    st.title("AI Chatbot")

    # Add customization options to the sidebar
    st.sidebar.title('Select an LLM')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length)

    user_question = st.text_area("Ask a question:")
    
    # Session state variable to hold chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    # Add a button to submit the question
    if st.button("Ask"):
        if user_question:
            response = conversation(user_question)
            message = {'human': user_question, 'AI': response['response']}
            st.session_state.chat_history.append(message)
            st.write("Chatbot:", response['response'])
        else:
            st.warning("Please enter a question before clicking 'Ask'.")

# Streamlit sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Employee Retention Prediction", "Work-Life Balance Preferences", "Recognition and Rewards", "AI Chatbot"])

# Display the selected page
if page == "Employee Retention Prediction":
    employee_retention_prediction()
elif page == "Work-Life Balance Preferences":
    work_life_balance_metrics()
elif page == "Recognition and Rewards":
    recognition_and_rewards()
else:
    chatbot()

def main():
    pass

if __name__ == "__main__":
    main()