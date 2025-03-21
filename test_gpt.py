from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=api_key)

# ID de tu trabajo de fine-tuning
job_id = "ftjob-Vh3Hb25bvpYmMXSbNlv3ZaJY"

def main():
    try:
        # Recuperar información del job
        job = client.fine_tuning.jobs.retrieve(job_id)
    except Exception as e:
        print(f"❌ Error al obtener la información del job:\n{e}")
        return

    print("✅ Información del trabajo:")
    print(f"ID: {job.id}")
    print(f"Estado: {job.status}")
    print(f"Modelo fine-tuned: {job.fine_tuned_model}")

    if job.status != "succeeded":
        print("\n⏳ El fine-tuning aún no ha finalizado o ha fallado.")
        return

    # Hacer una prueba con el modelo fine-tuned
    print("\n🤖 Probando el modelo fine-tuned...\n")

    try:
        response = client.chat.completions.create(
            model=job.fine_tuned_model,
            messages=[
                {"role": "user", "content": "Tengo alergia al gluten y quiero una dieta vegana para mantener mi peso. Mido 1.70 y peso 65 kg."}
            ],
            temperature=0.7
        )

        print("✅ Respuesta del modelo:")
        print(response.choices[0].message.content.strip())

    except Exception as e:
        print(f"❌ Error al hacer la solicitud al modelo fine-tuned:\n{e}")

if __name__ == "__main__":
    main()