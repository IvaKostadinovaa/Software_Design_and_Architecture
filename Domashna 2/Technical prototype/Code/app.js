// Import necessary libraries
const express = require('express');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();

// Create an Express app
const app = express();

// Connect to SQLite database (ensure the path is correct)
const db = new sqlite3.Database(path.join(__dirname, 'database', 'stock_data.db'), (err) => {
    if (err) {
        console.error('Error connecting to the SQLite database:', err.message);
    } else {
        console.log('Connected to the SQLite database.');
    }
});

// Serve static files from the "public" folder
app.use(express.static(path.join(__dirname, 'public')));

// API route to provide all stock data as JSON
app.get('/api/stock-data', (req, res) => {
    db.all('SELECT Код_на_издавач, Датум, Цена_на_последна_трансакција FROM stock_data ORDER BY Датум ASC', [], (err, rows) => {
        if (err) {
            console.error('Error fetching stock data:', err.message);
            res.status(500).send('Internal Server Error');
        } else {
            // Group data by company
            const groupedData = rows.reduce((acc, row) => {
                if (!acc[row.Код_на_издавач]) {
                    acc[row.Код_на_издавач] = { Код_на_издавач: row.Код_на_издавач, data: [] };
                }
                acc[row.Код_на_издавач].data.push({
                    Датум: row.Датум,
                    Цена_на_последна_трансакција: row.Цена_на_последна_трансакција,
                });
                return acc;
            }, {});

            res.json(Object.values(groupedData));
        }
    });
});

// API route to get a specific company's data
app.get('/api/stock-data/:company', (req, res) => {
    const companyName = req.params.company;
    db.all('SELECT Датум, Цена_на_последна_трансакција FROM stock_data WHERE Код_на_издавач = ? ORDER BY Датум ASC', [companyName], (err, rows) => {
        if (err) {
            console.error(`Error fetching data for ${companyName}:`, err.message);
            res.status(500).send('Internal Server Error');
        } else if (rows.length === 0) {
            res.status(404).send('Company not found');
        } else {
            res.json(rows);
        }
    });
});

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
