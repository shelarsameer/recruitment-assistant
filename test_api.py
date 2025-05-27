import requests
import json
import time

def test_api():
    # Test data
    test_data = {
        "jd": "Looking for a Python developer with experience in Flask and machine learning. Must have strong problem-solving skills and experience with REST APIs.",
        "resumes": [
            "Python developer with 3 years of experience in Flask and machine learning. Strong problem-solving skills and REST API development.",
            "Java developer with 5 years of experience in Spring Boot and microservices.",
            "Full-stack developer with experience in Python, JavaScript, and cloud technologies."
        ]
    }

    print("Testing Resume Matching API...")
    print("\nTest Job Description:")
    print(test_data["jd"])
    print("\nTest Resumes:")
    for i, resume in enumerate(test_data["resumes"]):
        print(f"\nResume {i+1}:")
        print(resume)

    # Make request to local API
    try:
        print("\nSending request to API...")
        response = requests.post(
            "http://localhost:5000/match",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("\n✅ API is working correctly!")
            results = response.json()
            print("\nTop Matches:")
            for match in results["top_matches"]:
                print(f"\nSimilarity Score: {match['similarity']:.2f}")
                print(f"Resume: {match['resume']}")
        else:
            print(f"\n❌ API returned status code: {response.status_code}")
            print("Response:", response.text)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to the API. Make sure the server is running.")
        print("Try running 'python app.py' in a separate terminal first.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    test_api() 