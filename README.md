# ChEMU Named Entity Recognition and Event Extraction

This project focuses on implementing and evaluating models for chemical entity recognition and event extraction from patents, based on the CLEF 2020 ChEMU (Chemistry Extraction from Molecular Utterances) dataset.

## Motivation

The extraction of structured information from chemical patents and literature is essential for:

1. **Scientific knowledge management**: Organizing chemical reactions and compounds in a structured database.
2. **Accelerating chemical research**: Enabling efficient search and discovery of chemical reactions and synthesis procedures.
3. **Supporting drug development**: Providing structured data for new molecule synthesis and chemical process optimization.
4. **Automating chemical knowledge extraction**: Reducing the manual effort needed to extract valuable information from patents.

This project explores how modern natural language processing techniques can be applied to automatically identify chemical entities and extract reaction events from scientific text, addressing the challenge of converting unstructured text into structured chemical knowledge.

## Dataset

The project uses the ChEMU dataset from CLEF 2020, which consists of:

- **Named Entity Recognition (NER) task**: Identifying chemical compounds and reaction conditions in patents.
- **Event Extraction (EE) task**: Extracting structured information about chemical reactions.

The dataset is provided in BRAT standoff format with .txt files containing the original text and corresponding .ann files containing annotations.

### Entity Types

The NER task includes identifying several types of entities:

- `EXAMPLE_LABEL`: Labels for examples in the patent
- `REACTION_PRODUCT`: The product of a reaction
- `STARTING_MATERIAL`: Initial compounds in a reaction
- `REAGENT_CATALYST`: Reagents and catalysts used
- `SOLVENT`: Solvents used in the reaction
- `OTHER_COMPOUND`: Other chemical compounds mentioned
- `TIME`: Time specifications
- `TEMPERATURE`: Temperature specifications
- `YIELD_OTHER`: Yield specifications in mass or moles
- `YIELD_PERCENT`: Percentage yield
- `REACTION_STEP`: A reaction step in a multi-step process
- `WORKUP`: Post-reaction workup steps

### Data Format

The dataset follows a standard format for NER tasks:
- Text files contain chemical patent excerpts describing reactions
- Annotation files identify entity spans and their types within the text

## Key Technical Concepts

### Model Architecture

This project explores several approaches for chemical entity recognition:

1. **BERT-based Models**: Using generic (e.g bert-base-uncased) and domain-specific models like ChemicalBERT for token classification, taking advantage of pre-trained chemical knowledge.

2. **Parameter-Efficient Fine-Tuning (PEFT)**:
   - Using LoRA (Low-Rank Adaptation) to efficiently fine-tune large language models with fewer trainable parameters.

### Technical Implementation Details

- **Data Processing Pipeline**: Converting BRAT annotations to BIO (Beginning, Inside, Outside) format for sequence labeling
- **Token-level Classification**: Using contextualized embeddings for named entity recognition
- **TensorBoard Integration**: For tracking training progress and visualizing model performance

## Results and Findings

- `High class imbalance`: When starting off, I've had pretty bad results and didn't noticed the problem immediately. When plotting the predictions, I realized that the model was predicting only the most frequent class. This was due to the high class imbalance in the dataset, where some classes had very few examples compared to others. To address this, I've used the `class_weights` parameter in the CrossEntropyLoss function, which helps the model pay more attention to underrepresented classes during training.
- `bert-base model vs chemical-bert`: surprisingly, `bert-base-uncased` shows better performance compared to the domain-specific models such as ChemicalBERT.
- `Classification head`: A shallow classification head with a single linear layer is sufficient for this task, as the BERT model already provides rich contextual embeddings. When experimenting with deeper classification heads, the training got more unstable and did not yield better results.
- `Classification head initialization`: Initializing the classification head with Normal distribution (e.g., `nn.init.normal_(mean=0, std=0.02)`) helps in stabilizing training and achieving better performance.
- `BIO vs IO`: Didn't explore this in detail, but annotating with BIO format decreases model performance.
- `Double Scheduler vs Single Scheduler`: I've started with a single scheduler, setting the learning rate to 4e-4 and got the results plotted to latter comparison. It had a not so fast convergence, specially in the beggining of the training, so I decided to try a double scheduler. I've maintained the same learning rate for the embedding model and used a higher learning rate for the classification head, which led to faster convergence in the beginning of the training. However, the final results were similar to the single scheduler approach.

## Citation

```
@incollection{he2020overview,
    author = {He, Jiayuan and Nguyen, Dat Quoc and Akhondi, Saber A. and Druckenbrodt, Christian and Thorne, Camilo and Hoessel, Ralph and Afzal, Zubair and Zhai, Zenan and Fang, Biaoyan and Yoshikawa, Hiyori and Albahem, Ameer and Cavedon, Lawrence and Cohn, Trevor and Baldwin, Timothy and Verspoor, Karin},
    title = {Overview of ChEMU 2020: Named Entity Recognition and Event Extraction of Chemical Reactions from Patents},
    booktitle = {Experimental IR Meets Multilinguality, Multimodality, and Interaction. Proceedings of the Eleventh International Conference of the CLEF Association (CLEF 2020)},
    publisher = {Lecture Notes in Computer Science},
    year = 2020,
    volume = 12260,
}
```

```
@misc{jadhav2024chemicalreactionextractionlong,
      title={Chemical Reaction Extraction from Long Patent Documents}, 
      author={Aishwarya Jadhav and Ritam Dutt},
      year={2024},
      eprint={2407.15124},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.15124}, 
}
```
