const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')
const abalone = require('./data/abalone.json')

function getCsvSize(filename) {
    const lines = fs.readFileSync(filename, 'utf-8').split(/\r?\n/);
    return {
        rows: lines.length - 1,
        columns: lines[0].split(',').length
    }
}

function sexToNumber(sex) {
    switch (sex){
        case 'F': return 0;
        case 'M': return 1;
        default: return 0.5
    }
}

function prepareData(filename) {
    const options = { hasHeader: true, columnConfigs: { rings: { isLabel: true } } };
    return tf.data.csv(`file://${filename}`, options).map(row => ({
        xs: Object.values(row.xs).map((x, i) => i === 0 ? sexToNumber(x) : x),
        ys: [row.ys.rings]
    })
    );
}

function createModel(inputShape, activation = 'sigmoid', lr = 0.01){
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape, activation, units: inputShape[0] * 2}));
    model.add(tf.layers.dense({units: 1}));
    model.compile({optimizer: tf.train.sgd(lr), loss: 'meanSquaredError'});
    return model
}

async function train({model, data, numRows, batchSize = 100, epochs = 200, trainRatio = .75}){
    const trainLength = Math.floor(numRows * trainRatio);
    const trainBatches = Math.floor(trainLength / batchSize);
    const shuffled = data.shuffle(100).batch(batchSize);
    const trainData = shuffled.take(trainBatches);
    const testData = shuffled.skip(trainBatches);
    await model.fitDataset(trainData, {epochs, validationData: testData})
}

//    [0,0.53,0.42,0.135,0.677,0.2565,0.1415,0.21,9],
//[1,0.44,0.365,0.125,0.516,0.2155,0.114,0.155,10],
//[0.5,0.33,0.255,0.08,0.205,0.0895,0.0395,0.055,7],

const tests = [
    [0,0.53,0.42,0.135,0.677,0.2565,0.1415,0.21],
[1,0.44,0.365,0.125,0.516,0.2155,0.114,0.155],
[0.5,0.33,0.255,0.08,0.205,0.0895,0.0395,0.055],

];

async function main(csvName){
    const data = prepareData(csvName);
    const size = getCsvSize(csvName);
    const model = createModel([size.columns - 1]);

    await train({model, data, numRows: size.rows});
    for (let i = 0; i < tests.length; i++){
        const test = tests[i];
        const output = model.predict(tf.tensor2d([test]));
        console.log(output.dataSync())
    }
}


const csvName = './data/abalone.csv';
main(csvName)
