import streamlit as st

st.title("Hallo! Meine Supply Chain App läuft.")
st.write("Wenn du das hier siehst, hat das Deployment geklappt!")

# Definieren des Textes mit HTML-Styling für die Namen
olga_robert_style = '<span style="font-weight: bold; font-size: 1.2em;">' # "1.2em" ist ca. 2 Nummern größer

st.markdown(f'''
Liebe {olga_robert_style}Olga</span> und cooler noch junger (aber auch nicht mehr so jung, wie es mal war) 
{olga_robert_style}Robert</span>. 
Das ist unsere Streamlit Oberfläche. Hier können wir unsere Daten visualisieren und interaktiv mit ihnen arbeiten. 
Ich freue mich schon darauf, gemeinsam mit dir die nächsten Schritte zu gehen und unsere App weiterzuentwickeln!

Hi, ich habs auch geschafft! lg Olga
''', unsafe_allow_html=True)