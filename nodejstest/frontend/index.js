const express = require("express");
const path = require("path");
const bodyParser = require("body-parser");
const { type } = require("os");
const spawn = require("child_process").spawn;

const app = express();

// Use bodyParser middleware to parse JSON request bodies
app.use(bodyParser.json());

// Define an API endpoint to receive data from the client
app.post("/api", (req, res) => {
    // Get the data sent from the client
    const clientData = req.body;

    console.log("Received data from client:", clientData);

    // Convert the data to an array for the Python script
    const commandArgs = Object.keys(clientData).map(key => `${key}=${clientData[key]}`);

    // Invoke the Python script with the data
    const path = require('path');
    const currentDirectory = path.dirname(__filename);
    console.log('Current directory:', currentDirectory);
    const dir = currentDirectory.substring(0, currentDirectory.lastIndexOf('\\'));
    const pythonProcess = spawn('python', [`${dir}/backend/main.py`, ...commandArgs]);

    pythonProcess.stdout.on('data', (data) => {
        // Convert Buffer data to string
        let dataString = data.toString().trim();

        try {
            // Parse the data from the Python script as JSON
            const jsonData = JSON.parse(dataString);
            
            console.log("Received data from Python script:", dataString);
            
            // Send the response back to the client
            res.json({ message: "Data processed successfully", data: dataString });
        } catch (error) {
            console.error("Error parsing Python script response as JSON:", error);
            res.status(500).json({ error: "Failed to parse response from Python script" });
        }
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error("Python script error:", data.toString());
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python script exited with code ${code}`);
    });
});

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve index.html from the 'public' directory
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
const port = 10000;
app.listen(port, () => {
    console.log(`Server started on port ${port}`);
});
