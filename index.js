// Requiring module
const express = require("express");
const app = express();
const path = require("path");
const PNG = require("pngjs").PNG;
const cors = require('cors');
app.use(cors({
    origin: 'http://localhost:2020',
    credentials: true,
    optionsSuccessStatus: 200
}));

let class_names = ['attribute', 'composition', 'inheritance', 'rectangle', 'reference'];

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

// Set public as static directory
app.use(express.static('public'));
app.set('views', path.join(__dirname, '/views'))

// Use ejs as template engine
app.set('view engine', 'ejs');

app.use(express.json({limit: '50mb'}));
app.use(express.urlencoded({limit: '50mb', extended: true}));


// Render main template
/*app.get('/', (req, res) => {
    res.render('main')
})*/
global.model = null;
// Setting up tfjs with the model we downloaded
tf.loadGraphModel('file:///users/cse-ga/PhpstormProjects/model_deploy/public/model/model.json')
    .then(function (model) {
        global.model = model;
        console.log("model loaded");
    });

app.post('/imageClassification', function (req, res) {
    let data = req.body.image;
    data = data.replace(/^data:image\/png+;base64,/, "");
    data = data.replace(/ /g, '+');

    let imageData = Buffer.from(data, 'base64');

    const bitmap = PNG.sync.read(imageData);
    let imgData = new Array(bitmap.height);

    for (let y = 0; y < bitmap.height; y++) {
        let row = [bitmap.width];
        for (let x = 0; x < bitmap.width * 4; x += 4) {
            row [x / 4] = bitmap.data[y * bitmap.width * 4 + x];
            if (bitmap.data[x] !== 255) {
                console.log(bitmap.data[x]);
            }
        }
        imgData[y] = row;
    }
    classifyImage(imgData).then(function (response){
        if(response != null) {
            res.send(response);
        }
    });
});

async function classifyImage(imgData) {
    var result = null;
    try {
        if (global.model != null) {
            result = await predict(imgData);
        } else {
            setTimeout(async function () {
                result = await predict(imgData);
            }, 100);
        }

    } catch (e) {
        return Promise.reject(e.message);
    }

     return Promise.resolve(result);
}

async function predict(input) {
    let predicted_class = " ";
    try {
        predicted_class = await global.model.predict([tf.tensor(input)
            .reshape([-1, 256, 256, 1])])
            .array().then(function (scores) {
            scores = scores[0];
            let predicted = scores
                .indexOf(Math.max(...scores));
            predicted_class = class_names[predicted];
            return predicted_class;
        });
        return predicted_class;
    }
    catch(e){
        return e;
    }
}

// Server setup
app.listen(3000, () => {
    console.log("The server started running on port 3000")
});