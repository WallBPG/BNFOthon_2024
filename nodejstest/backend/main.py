import argparse
import sys
import json

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Combine two strings with a space in the middle.")
    
    # Add two positional arguments: arg1 and arg2
    parser.add_argument("age")
    parser.add_argument("sex")
    parser.add_argument("pneumonia")
    parser.add_argument("pregnant")
    parser.add_argument("diabetes")
    parser.add_argument("copd")
    parser.add_argument("asthma")
    parser.add_argument("inmsupr")
    parser.add_argument("hipertension")
    parser.add_argument("otherDisease")
    parser.add_argument("cardiovascular")
    parser.add_argument("obesity")
    parser.add_argument("renalChronic")
    parser.add_argument("tobacco")
    # Parse the arguments
    args = parser.parse_args()
    
    args_dict = {
        "age": args.age,
        "sex": args.sex,
        'pneumonia': args.pneumonia,
        'pregnant': args.pregnant,
        'diabetes': args.diabetes,
        'copd': args.copd,
        'asthma': args.asthma,
        'inmsupr': args.inmsupr,
        'hipertension': args.hipertension,
        'otherDisease': args.otherDisease,
        'cardiovascular': args.cardiovascular,
        'obesity': args.obesity,
        'renalChronic': args.renalChronic,
        'tobacco': args.tobacco,
        }
    #use args_dict to calculate score###########################


    #for testing
    score_dict = {
        "score": 50,
        }
    json_data = json.dumps(score_dict)
    # Print the json data
    print(json_data)
    sys.stdout.flush()

if __name__ == "__main__":
    main()

