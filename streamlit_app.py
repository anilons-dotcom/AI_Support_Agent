import streamlit as st
#from rag_agent import ask_rag, remember_fact, remember_preference, build_pdf_index, build_email_index
from rag_agent import ask_rag, remember_fact, remember_preference, build_pdf_index
st.set_page_config(page_title="Hybrid RAG Agent", layout="wide")

# Initialize indexes once
if "indexes_built" not in st.session_state:
    with st.spinner("Building PDF and Email indexes..."):
        build_pdf_index()
        #build_email_index()
    st.session_state.indexes_built = True

# Title
st.title("📄📧 Hybrid RAG Support Agent")
st.markdown("Ask questions based on your PDFs, CSVs, and Outlook emails. Memory-aware, privacy-first.")

# Input
user_input = st.text_input("Your question", placeholder="Ask anything...")

# Submit
if st.button("Submit") and user_input:
    if "remember that" in user_input.lower():
        fact = user_input.split("remember that", 1)[1].strip()
        remember_fact(fact)
        st.success("✅ Fact remembered.")
    elif "my preference is" in user_input.lower():
        pref = user_input.split("my preference is", 1)[1].strip()
        if "=" in pref:
            key, value = pref.split("=", 1)
            remember_preference(key.strip(), value.strip())
            st.success("✅ Preference saved.")
        else:
            st.warning("⚠️ Format should be: my preference is key=value")
    else:
        with st.spinner("Thinking..."):
            response = ask_rag(user_input)
        st.markdown("### 💬 Answer")
        st.write(response)

# Footer
st.markdown("---")
st.caption("Built with LlamaIndex + Ollama · Memory-aware · Local-first · EU-compliant")