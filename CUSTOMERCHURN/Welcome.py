import streamlit as st

st.title("CUSTOMER CHURN PREDICTOR")
st.write("Welcome!")
st.write("Customer churn is more than a lost customer — it can be a signal that something in the customer experience needs attention.")

st.write('**Customer Churn Predictor** uses machine learning to estimate whether a customer is likely to churn based on factors such as usage behavior, support interactions, payment delays, tenure, spending, and recent activity.')



st.write('**🔮 Predict Churn**')
st.write('Enter customer information and receive a churn prediction along with the probability of churn.')

st.write('**⚖️ Compare Models**')
st.write('Compare **Logistic Regression** and **Random Forest** to understand how different machine-learning approaches perform on the same problem.')

st.write('**🧠 Understand the Prediction**')
st.write('Use **SHAP explainability** to see which customer features pushed the model toward or away from a churn prediction.')


st.write('This isn t just a yes/no prediction.')

st.write('The goal is to answer three questions:')

st.markdown('**1. Is this customer likely to churn?**')
st.write('Get a probability-based prediction instead of relying only on a binary label.')

st.write('**2. What influenced the prediction?**')
st.write('Identify the features that contributed most strongly to the model s decision.')

st.write('**3. Which model performs better?**')
st.write("Evaluate different approaches and understand their strengths rather than treating one model as automatically superior.")
st.write('**Python • Scikit-learn • Pandas • SHAP • Streamlit**')
st.write('Choose a section from the navigation or start with a customer prediction.')

if st.button("PREDICT") :
    st.switch_page("pages/project.py")