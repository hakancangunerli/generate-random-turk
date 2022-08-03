import streamlit as st 
from main import main
# add a button to the page

st.title("Generate a random turk")
buttonclick = st.button("Generate a random turk")
main()
#when button is clicked run the main function and display the result which is 'img.png
if buttonclick == True:
    st.write("Here is your random turk")
    st.image("img.png")
else: 
    st.write("You didn't click the button or there was an error while generating the random turk")
    