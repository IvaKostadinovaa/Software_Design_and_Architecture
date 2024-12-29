const express = require('express');
const path = require('path');
const sqlite3 = require('sqlite3').verbose(); // Add SQLite library

const app = express();

const db = new sqlite3.Database(path.join(__dirname, 'database', 'stock_data.db'), (err) => {
    if (err) {
        console.error('Error connecting to the SQLite database:', err.message);
    } else {
        console.log('Connected to the SQLite database.');
    }
});

app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/stock-data', (req, res) => {
    const query = `SELECT
                       Код_на_издавач,
                       Датум,
                       Цена_на_последна_трансакција,
                       Мак_,
                       Мин_,
                       Просечна_цена,
                       Промет_во_БЕСТ_во_денари,
                       Вкупен_промет_во_денари,
                       Количина,
                       Промет_во_БЕСТ_во_денари_друга
                   FROM stock_data
                   ORDER BY Датум ASC`;

    db.all(query, [], (err, rows) => {
        if (err) {
            console.error('Error fetching stock data:', err.message);
            res.status(500).send('Internal Server Error');
        } else {
            res.json(rows);
        }
    });
});


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

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
