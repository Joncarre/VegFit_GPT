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

# ID of your fine-tuning job
job_id = "ftjob-Vh3Hb25bvpYmMXSbNlv3ZaJY"

def main():
    try:
        # Retrieve job information
        job = client.fine_tuning.jobs.retrieve(job_id)
    except Exception as e:
        print(f"❌ Error getting job information:\n{e}")
        return

    print("✅ Job Information:")
    print(f"ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Fine-tuned model: {job.fine_tuned_model}")

    if job.status != "succeeded":
        print("\n⏳ Fine-tuning hasn't finished or has failed.")
        return

    # Test the fine-tuned model
    print("\n🤖 Testing the fine-tuned model...\n")

    try:
        response = client.chat.completions.create(
            model=job.fine_tuned_model,
            messages=[
                {"role": "user", "content": "I have gluten allergy and want a vegan diet to maintain my weight. I'm 1.70m tall and weigh 65kg."}
            ],
            temperature=0.7
        )

        print("✅ Model response:")
        print(response.choices[0].message.content.strip())

    except Exception as e:
        print(f"❌ Error making request to fine-tuned model:\n{e}")

if __name__ == "__main__":
    main()