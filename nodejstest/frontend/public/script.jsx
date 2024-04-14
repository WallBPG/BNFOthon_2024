//References
let quizContainer = document.getElementById("container");
let nextBtn = document.getElementById("next-button");
let countOfQuestion = document.querySelector(".number-of-question");
let displayContainer = document.getElementById("display-container");
let scoreContainer = document.querySelector(".score-container");
let restart = document.getElementById("restart");
let userScore = document.getElementById("user-score");
let startScreen = document.querySelector(".start-screen");
let startButton = document.getElementById("start-button");
let questionCount;


//for Testing
let riskIndex = .68;

//Questions and Options array

const quizArray = [
    {
        id: "0",
        question: "Enter Age",
        options: "int",
        key: "age",
    },
    {
        id: "1",
        question: "Enter Sex",
        options: ["Male", "Female"],
        key: "sex",
    },
    {
        id: "2",
        question: "Are You Pregnant?",
        options: ["Yes", "No", "Unknown"],
        key: "pregnant",
    },
    {
        id: "3",
        question: `Do You Have Chronic Obstructive Pulmonary Disease (COPD)?
        <i class="bi bi-question-circle-fill info" data-bs-toggle="modal" data-bs-target="#COPDModal" onmouseover="" style="cursor: pointer;"></i> 
        `,
        options: ["Yes", "No", "Unknown"],
        key: "copd",
    },
    {
        id: "4",
        question: "Do You Have Asthma?",
        options: ["Yes", "No", "Unknown"],
        key: "asthma",
    },
    {
        id: "5",
        question: "Do You Have Pneumonia?",
        options: ["Yes", "No", "Unknown"],
        key: "pneumonia",
    },
    {
        id: "6",
        question: "Do You Have Diabetes?",
        options: ["Yes", "No", "Unknown"],
        key: "diabetes",
    },
    {
        id: "7",
        question: `Are You Immunosuppressed?
        <i class="bi bi-question-circle-fill info" data-bs-toggle="modal" data-bs-target="#immuneModal" onmouseover="" style="cursor: pointer;"></i>
        `,
        options: ["Yes", "No", "Unknown"],
        key: "inmsupr",
    },
    {
        id: "8",
        question: `Do You Have Hypertension?
        <i class="bi bi-question-circle-fill info" data-bs-toggle="modal" data-bs-target="#hyperModal" onmouseover="" style="cursor: pointer;"></i>
        `,
        options: ["Yes", "No", "Unknown"],
        key: "hipertension",
    },
    {
        id: "9",
        question: `Do You Have a Heart or Blood Vessel Related Disease?
        <i class="bi bi-question-circle-fill info" data-bs-toggle="modal" data-bs-target="#heartModal" onmouseover="" style="cursor: pointer;"></i>`,
        options: ["Yes", "No", "Unknown"],
        key: "cardiovascular",
    },
    {
        id: "10",
        question: `Do You Have Chronic Renal Disease?
        <i class="bi bi-question-circle-fill info" data-bs-toggle="modal" data-bs-target="#renalModal" onmouseover="" style="cursor: pointer;"></i>`,
        options: ["Yes", "No", "Unknown"],
        key: "renalChronic",
    },
    {
        id: "11",
        question: `Are You Obese?
          <i class="bi bi-question-circle-fill info" data-bs-toggle="modal" data-bs-target="#obeseModal" onmouseover="" style="cursor: pointer;"></i>`,
        options: ["Yes", "No", "Unknown"],
        key: "obesity",
    },
    {
        id: "12",
        question: "Do You Use Tobacco?",
        options: ["Yes", "No", "Unknown"],
        key: "tobacco",
    },
    {
        id: "13",
        question: "Do You Have Another Disease?",
        options: ["Yes", "No", "Unknown"],
        key: "otherDisease",
    },
];

let responses = {};

//Restart Quiz
restart.addEventListener("click", () => {
    initial();
    displayContainer.classList.remove("hide");
    scoreContainer.classList.add("hide");
});

//Next Button
nextBtn.addEventListener(
    "click",
    (displayNext = () => {
        //increment questionCount
        questionCount += 1;
        if (questionCount == quizArray.length-1){
            nextBtn.innerText = `Calculate`
        }
        //if last question
        if (questionCount == quizArray.length) {
            //hide question container and display score
            displayContainer.classList.add("hide");
            scoreContainer.classList.remove("hide");
            //user risk index
            userScore.innerHTML =
                `Your risk index is ${riskIndex} <i class="bi bi-question-circle-fill" data-bs-toggle="modal" data-bs-target="#riskModal" onmouseover="" style="cursor: pointer;"></i>`;
        } else {
            //display questionCount
            countOfQuestion.innerHTML =
                questionCount + 1 + " of " + quizArray.length + " Question";
            //display quiz
            quizDisplay(questionCount);
        }
    })
);


//Display quiz
const quizDisplay = (questionCount) => {
    let quizCards = document.querySelectorAll(".container-mid");
    nextBtn.setAttribute("disabled",true);
    //Hide other cards
    quizCards.forEach((card) => {
        card.classList.add("hide");
    });
    //display current question card
    quizCards[questionCount].classList.remove("hide");

    if (typeof jQuery !== "undefined") {
        console.log("jQuery is loaded and ready to use.");
    } else {
        console.log("jQuery is not loaded.");
    }

    // Event handler for age input change
    $("#age").on('input', function() {
        console.log('Age input changed, value:', $(this).val());
        if ($(this).val() !== ""){
          $("#next-button").prop("disabled", false);
        }
        else{
          $("#next-button").prop("disabled", true);
        }
        
    });

    // Event handler for radio button selection
    $("input[type=radio]").on('change', function() {
        console.log('Radio button changed, selected:', $(this).val());
        $("#next-button").prop("disabled", false);
    });
};
    

//Quiz Creation
function quizCreator() {
    //generate quiz
    for (let i of quizArray) {
        //quiz card creation
        let div = document.createElement("div");
        div.classList.add("container-mid", "hide", "flex-column");
        div.setAttribute("data-toggle", "buttons");
        //question number
        countOfQuestion.innerHTML = 1 + " of " + quizArray.length + " Question";
        //question
        let question_DIV = document.createElement("p");
        question_DIV.classList.add("question");
        question_DIV.innerHTML = i.question;
        div.appendChild(question_DIV);
        //options
        if (i.options == "int") {
          div.innerHTML += `
          <input type="number" id="age" name="age" min="0" value = "0">
          `;
        }
        else {
          for (let j = 0; j < i.options.length; j++){
            div.innerHTML += `
            <div class="card my-1">
              <div class="card-body p-0">
                <label class="btn btn-default blue w-100 py-3">
                  <input type="radio" class="toggle " name= "toggle" value="${i.options[j]}"> ${i.options[j]}
                </label>
              </div>
            </div>
            `;             
          }
        }
        quizContainer.appendChild(div);
    }
}

function uncheckAllRadioButtons() {
    const radioButtons = document.querySelectorAll('input[type="radio"]');
    radioButtons.forEach((radio) => {
        radio.checked = false;
    });
}

//function to record user choices
function checker() {
    let userChoice
    if (questionCount==0){
      userChoice = Number($('input[id=age]').val());
      responses.age = userChoice
    }
    else{
      userChoice = $('input[name=toggle]:checked').val();
      switch (userChoice) {
        case 'Yes':
            userChoice = 1;
            break;
        case 'No':
            userChoice = 0;
            break;
        case 'Unknown':
            userChoice = 2;
            break;
        }
        let keyword = quizArray[questionCount].key
        responses[keyword] = userChoice
    }
    if (questionCount == quizArray.length-1){
      toIndexJs(responses);
      //test values
      //toIndexJs({age: 20, sex: 0})
    }
    
  
    console.log(responses);
    uncheckAllRadioButtons();
}

//initial setup
function initial() {
    quizContainer.innerHTML = "";
    nextBtn.innerText = `Next`
    questionCount = 0;
    scoreCount = 0;
    quizCreator();
    quizDisplay(questionCount);
}

//when user click on start button
startButton.addEventListener("click", () => {
    responses = [];
    startScreen.classList.add("hide");
    displayContainer.classList.remove("hide");
    initial();
});

//hide quiz and display start screen
window.onload = () => {
    startScreen.classList.remove("hide");
    displayContainer.classList.add("hide");
};

function toIndexJs(data = {}) {
    // Use Fetch API to send data to the server
    fetch('/api', {
        method: 'POST', // Use POST method to send data
        headers: {
            'Content-Type': 'application/json' // Specify content type as JSON
        },
        body: JSON.stringify(data) // Convert data to JSON string
    })
    .then(response => response.json()) // Parse JSON response
    .then(data => {
        console.log("Received data from server:", data);
        // Parse the data field to convert it from a JSON string to an object
        data = JSON.parse(data.data);
        
        // Access the `score` property
        riskIndex = parseFloat(data.score);
        console.log(riskIndex)
        userScore.innerHTML =
        `Your risk index is ${riskIndex}% <i class="bi bi-question-circle-fill" data-bs-toggle="modal" data-bs-target="#riskModal" onmouseover="" style="cursor: pointer;"></i>`;
        //tempScore = 50
        //riskIndex = parseFloat(tempScore); // Convert the stripped score to a number
        })
    .catch(error => {
        console.error("Error sending data to server:", error);
    });
}
