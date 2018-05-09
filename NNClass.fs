namespace NNClass
open System;
open System.IO;

type NeuralNetwork (numInput: int, numHidden: int, numOutput: int, seed: int) =
    let numWeights = 
        (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

    let inputs = Array.create numInput 0.0;
    let mutable ihWeights = Array.init numInput (fun r -> Array.create numHidden 0.0)
    let hBiases = Array.create numHidden 0.0;
    let hOutputs = Array.create numHidden 0.0;

    let hoWeights = Array.init numHidden (fun r -> Array.create numOutput 0.0)
    let oBiases = Array.create numOutput 0.0;
    let outputs = Array.create numOutput 0.0;

    let rnd = new Random (seed); // used by randomize and by train/shuffle

    member this.GetWeights (): double[] =
        let result = Array.create numWeights 0.0;
        let mutable k = 0;

        for i = 0 to ihWeights.Length-1 do
            for j = 0 to ihWeights.[i].Length-1 do
                result.[k] <- ihWeights.[i].[j];
                k <- k + 1
        for i = 0 to hBiases.Length-1 do
            result.[k] <- hBiases.[i];
            k <- k + 1
        for i = 0 to hoWeights.Length-1 do
            for j = 0 to hoWeights.[i].Length-1 do
                result.[k] <- hoWeights.[i].[j];
                k <- k + 1
        for i = 0 to oBiases.Length-1 do
            result.[k] <- oBiases.[i];
            k <- k + 1
        result;
        
    member this.SetWeights (weights: double[]) =
        // copy serialized weights and biases in weights[] array
        // to i-h weights, h biases, h-o weights, o biases
        if weights.Length <> numWeights then
            raise (new Exception ("Bad weights array in SetWeights"));

        let mutable k = 0; // points into weights param

        for i = 0 to numInput-1 do
            for j = 0 to numHidden-1 do
                ihWeights.[i].[j] <- weights.[k];
                k <- k + 1
        for i = 0 to numHidden-1 do
            hBiases.[i] <- weights.[k];
            k <- k + 1
        for i = 0 to numHidden-1 do
            for j = 0 to numOutput-1 do
                hoWeights.[i].[j] <- weights.[k];
                k <- k + 1
        for i = 0 to numOutput-1 do
            oBiases.[i] <- weights.[k];
            k <- k + 1
        
    member this.XavierWeights () = 
        // initialize weights and biases approx. Xavier
        // ***
        // this.SetWeights (...);
        ()
        
    member this.RandomiseWeights () = 
        // initialize weights and biases to small random values
        let initFun i = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001
        let initialWeights = Array.init numWeights (initFun);
        this.SetWeights (initialWeights);
        
    static member LoadModel (modelName: string, nnSeed:int) =
        let lines = 
            File.ReadAllLines (modelName)
            |> Array.map (fun line -> 
                let c = line.IndexOf ("//")
                if c >= 0 then line.Substring (0, c)
                else line)
            |> Array.filter (fun line -> line.Trim() <> "")
        
        let nums =
            lines.[0].Split(' ')
            |> Array.collect (fun tok -> 
                if tok.Trim() <> "" then [| Int32.Parse (tok) |] else [| |]) 
        
        let numInput, numHidden, numOutput = nums.[0], nums.[1], nums.[2]
        //printfn "nums: %A" (numInput, numHidden, numOutput)
        
        let weights = 
            lines.[1..]
            |> Array.collect (fun line -> 
                line.Split(' ')
                |> Array.collect (fun tok -> 
                    if tok.Trim() <> "" then [| Double.Parse (tok) |] else [| |]
                    )
            )
        //printfn "weights (%d): %A" weights.Length weights
        
        let nn = NeuralNetwork (numInput, numHidden, numOutput, nnSeed);
        nn.SetWeights (weights);
        nn
      
    member this.SaveModel (modelName: string) =
        use tw = File.CreateText (modelName);
        this.FPrintModel (tw)
        
    member this.FPrintModel (tw: TextWriter) =

        fprintfn tw "// numInput numHidden numOutput\n"
        fprintfn tw "%d %d %d\n" numInput numHidden numOutput
        fprintfn tw "\n// i-h weights (4*5)://"

        for i=0 to ihWeights.Length-1 do
            for j=0 to ihWeights.[0].Length-1 do
                fprintf tw "%0.4f " ihWeights.[i].[j]
            fprintfn tw "\n"

        fprintfn tw "\n// h biases (5):"

        for i=0 to hBiases.Length-1 do
            fprintf tw "%0.4f " hBiases.[i]

        fprintfn tw "\n// h-o weights (5*3):"

        for i=0 to hoWeights.Length-1 do
            for j=0 to hoWeights.[0].Length-1 do
                fprintf tw "%0.4f " hoWeights.[i].[j]
            fprintfn tw "\n"

        fprintfn tw "\n // o biases (3):"
        for i=0 to oBiases.Length-1 do
            fprintf tw "%0.4f " oBiases.[i]

//        Array.create numWeights 0.0 |> Array.iter (fun v -> fprintf tw "%f " v)
        fprintfn tw "\n//"
        ()

    member this.HyperTanFunction (x: double): double =
        if x < -20.0 then -1.0; // approximation is correct to 30 decimals
        elif x > 20.0 then 1.0;
        else Math.Tanh (x);

    member this.Softmax (oSums: double[]): double[] =
        // determine max output sum
        // does all output nodes at once so scale doesn't have to be re-computed each time
        
        let sum = Array.sumBy (fun v -> Math.Exp(v)) oSums;
        let result = Array.map (fun v -> Math.Exp(v) / sum) oSums;
        result; // now scaled so that xi sum to 1.0

    member this.Shuffle (sequence: int[]) = // Fisher
        for i = 0 to sequence.Length-1 do
            let r = rnd.Next (i, sequence.Length);
            let tmp = sequence.[r];
            sequence.[r] <- sequence.[i];
            sequence.[i] <- tmp;

    member this.ComputeOutputs (xValues: double[]): double[] =
        let hSums = Array.create numHidden 0.0 // hidden nodes sums scratch array
        let oSums = Array.create numOutput 0.0; // output nodes sums

        for i in 0 .. xValues.Length-1 do // copy x-values to inputs
            inputs.[i] <- xValues.[i];
        // note: no need to copy x-values unless you implement a ToString.
        // more efficient is to simply use the xValues[] directly.

        for j in 0 .. numHidden-1 do  // compute sum of (ia) weights * inputs
            for i in 0 .. numInput-1 do
                hSums.[j] <- hSums.[j] + inputs.[i] * ihWeights.[i].[j]; 

        for i in 0 .. numHidden-1 do  // add biases to a sums
            hSums.[i] <- hSums.[i] + hBiases.[i];

        for i in 0 .. numHidden-1 do   // apply activation
            hOutputs.[i] <- this.HyperTanFunction (hSums.[i]); // hard-coded

        for j in 0 .. numOutput-1 do  // compute h-o sum of weights * hOutputs
            for i in 0 .. numHidden-1 do
                oSums.[j] <- oSums.[j] + hOutputs.[i] * hoWeights.[i].[j]; 

        for i in 0 .. numOutput-1 do  // add biases to input-to-hidden sums
            oSums.[i] <- oSums.[i] + oBiases.[i];

        let softOut = this.Softmax (oSums); // all outputs at once for efficiency
        Array.blit softOut 0 outputs 0 softOut.Length;

        let retResult = Array.copy outputs; // could define a GetOutputs method
        retResult;

    member this.Error (trainData: double[][]): double =
        // average squared error per training item
        let mutable sumSquaredError = 0.0;

        let xValues = Array.create numInput 0.0 //first numInput values in trainingData
        let tValues = Array.create numOutput 0.0 //last numOutput values

        //Iterating through each training case:
        for i=0 to trainData.Length-1 do 
            Array.blit trainData.[i] 0 xValues 0 numInput
            Array.blit trainData.[i] numInput xValues 0 numOutput
            let yvalues = this.ComputeOutputs(xValues);
            for j=0 to numOutput-1 do
                let mutable err = tValues.[j] - yvalues.[j]
                sumSquaredError <- sumSquaredError + (err * err)
        sumSquaredError/(float trainData.Length)

    member this.MaxIndex (vector: double[]): int = // helper for Accuracy()
        // index of largest value
      let mutable bigIndex = 0;
      let mutable biggestVal = vector.[0]; //while assigning for the first time, we use '=' operator.
      for i=0 to vector.Length-1 do
         if vector.[i] > biggestVal then
            biggestVal <- vector.[i]
            bigIndex <- i 
      bigIndex


    member this.Accuracy (testData: double[][]): double =
        // percentage correct using winner-takes all

        let mutable numCorrect = 0;
        let mutable numWrong = 0;

        let mutable xValues = Array.create numInput 0.0
        let mutable tValues = Array.create numOutput 0.0
        let mutable yValues = Array.empty<float>

        for i=0 to testData.Length-1 do 
             Array.blit testData.[i] 0 xValues 0 numInput
             Array.blit testData.[i] numInput tValues 0 numOutput
             yValues <- this.ComputeOutputs xValues
             let mutable maxIndex = this.MaxIndex yValues
             let mutable tMaxIndex = this.MaxIndex tValues

             if maxIndex = tMaxIndex then
                 numCorrect <- numCorrect + 1
             
             else
                 numWrong <- numWrong + 1

        ((float numCorrect) * 1.0)/ ((float numCorrect) + (float numWrong))


   
    member this.Train (trainData: double[][], maxEpochs: int, learnRate: double, momentum: double, errtw: TextWriter): double[] =
      try
        //Training using back-prop algorithm

        //back-prop specific arrays
        let hoGrads = Array.init numHidden (fun r -> Array.create numOutput 0.0) //hidden-to-output weights gradients
        let obGrads = Array.create numOutput 0.0 // output biases gradients

        let ihGrads = Array.init numInput (fun r -> Array.create numHidden 0.0) //input-to-hidden weights gradients
        let mutable hbGrads = Array.create numHidden 0.0 //hidden biases Gradients

        let mutable oSignals = Array.create numOutput 0.0 //output signals - gradients w/o associated input terms
        let hSignals = Array.create numHidden 0.0 // hidden node signals

        //back-prop momentum specific arrays
        let ihPrevWeightsDelta = Array.init numInput (fun r -> Array.create numHidden 0.0)
        let mutable hPrevBiasesDelta = Array.create numHidden 0.0
        let hoPrevWeightsDelta = Array.init numHidden (fun r -> Array.create numOutput 0.0)
        let mutable oPrevBiasesDelta = Array.create numOutput 0.0

        //train a back-prop style Neural Network classifier using learning rate and momentum
        let mutable epoch = 0
        let mutable xValues = Array.create numInput 0.0 //inputs (in 'double' datatype)
        let mutable tValues = Array.create numOutput 0.0 //target values (in 'double' datatype)

        let mutable sequence = Array.create trainData.Length 0 //its an integer

        for i = 0 to sequence.Length-1 do 
            sequence.[i] <- i

        let mutable errInterval = maxEpochs/10 //interval to check validation data

        while epoch < maxEpochs do
            epoch <- epoch+1

            // creating a Log for errors during the training process.
            let trainErr = this.Error(trainData)
            fprintfn errtw "%d %s" epoch (trainErr.ToString("F4"))

            if epoch % errInterval = 0 && epoch< maxEpochs then
            //if progress = true && epoch % errInterval = 0 && epoch< maxEpochs then
                let mutable trainErr = this.Error(trainData)
                printfn "epoch = %d training Error %s" epoch (trainErr.ToString("F4"))
            
            this.Shuffle(sequence) //visiting each training data in random order

            for ii = 0 to trainData.Length-1 do
                let idx = sequence.[ii]

                Array.blit trainData.[idx] 0 xValues 0 numInput
                Array.blit trainData.[idx] numInput tValues 0 numOutput

                let mutable compOutputs = this.ComputeOutputs(xValues) //copy xValues in computeValues

                //indices = i, j= hidden, k = outputs

                //1.Compute output nodes signals (assumes softmax)
                for k=0 to numOutput-1 do
                    oSignals.[k] <- (tValues.[k] - outputs.[k]) * (1.0- outputs.[k]) * outputs.[k]

                //2.Compute Hidden-to-Output weights gradients using Output Signals
                for j=0 to numHidden-1 do
                    for k=0 to numOutput-1 do
                        hoGrads.[j].[k] <- (oSignals.[k] * hOutputs.[j])
                
                //2b. Compute output Biases gradients using output signals
                for k=0 to numOutput-1 do
                    obGrads.[k] <- oSignals.[k] * 1.0 //dummy assoc. input value

                //3. Compute hidden nodes signals
                for j=0 to numHidden-1 do
                    let mutable sum = 0.0 // need sums of output signals times hidden-to-output weights
                    for k=0 to numOutput-1 do
                        sum <- sum + (oSignals.[k] * hoWeights.[j].[k])
                    hSignals.[j] <- (1.0 + hOutputs.[j]) * (1.0 - hOutputs.[j]) * sum //assumes tanh

                //4. Compute input-hidden weights gradients
                for i=0 to numInput-1 do
                    for j=0 to numHidden-1 do
                        ihGrads.[i].[j] <- hSignals.[j] * inputs.[i]

                //4b. Compute hidden node biases gradients
                for j=0 to numHidden-1 do
                    hbGrads.[j] <- hSignals.[j] * 1.0 //dummy 1.0 input

                //Now, Update weights and biases

                //update input-to-hidden weights
                for i=0 to numInput-1 do
                    for j=0 to numHidden-1 do
                        let delta = ihGrads.[i].[j] * learnRate
                        ihWeights.[i].[j] <- ihWeights.[i].[j] + delta
                        ihWeights.[i].[j] <- ihWeights.[i].[j] + (ihPrevWeightsDelta.[i].[j] * momentum)
                        ihPrevWeightsDelta.[i].[j] <- delta //save for next time

                //update Hidden biases
                for j=0 to numHidden-1 do
                    let delta = hbGrads.[j] * learnRate
                    hBiases.[j] <- hBiases.[j] + delta
                    hBiases.[j] <- hBiases.[j] + (hPrevBiasesDelta.[j] * momentum)
                    hPrevBiasesDelta.[j] <- delta

                //update hidden-to-output weights
                for j=0 to numHidden-1 do
                    for k=0 to numOutput-1 do
                        let delta = hoGrads.[j].[k] * learnRate
                        hoWeights.[j].[k] <- hoWeights.[j].[k] + delta
                        hoWeights.[j].[k] <- hoWeights.[j].[k] + hoPrevWeightsDelta.[j].[k] * momentum
                        hoPrevWeightsDelta.[j].[k] <- delta


                //update Output node biases
                for k=0 to numOutput-1 do
                    let delta = obGrads.[k] * learnRate
                    oBiases.[k] <- oBiases.[k]+delta
                    oBiases.[k] <- oBiases.[k]+(oPrevBiasesDelta.[k] * momentum)
                    oPrevBiasesDelta.[k] <- delta

        //
        
        let bestWeights = this.GetWeights ();
        bestWeights;
        
      with ex -> 
          Console.WriteLine ("*** {0}", ex.StackTrace);
          [| 0.0 |]

// ---
