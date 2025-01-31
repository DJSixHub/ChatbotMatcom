import streamlit as st
from app import Chat

user_chat = []
bot_chat = []

def generar_interfaz(nombre, chat_instance):
    st.header(f"{nombre}")

    user_input = st.text_input("Ingrese su consulta:")
    if user_input:
        user_chat.append(user_input)
        bot_response = chat_instance.Run(user_input)
        bot_chat.append(bot_response)
    
    for i in range(len(user_chat)):
        st.markdown(f"**Usuario:** {user_chat[i]}")
        st.markdown(f"**Asistente:** {bot_chat[i]}")

def main():
    st.title("Bienvenido a su asistente virtual")
    st.markdown("---")

    option = st.selectbox("Seleccione una opción:", ["", "Asistente Vocacional", "Asistente Jurídico"])

    chat_instance = Chat()

    if option == "Asistente Vocacional":
        generar_interfaz("Asistente Vocacional", chat_instance)
        
    elif option == "Asistente Jurídico":
        generar_interfaz("Asistente Jurídico", chat_instance)


if __name__ == "__main__":
    main()
