# Qualitative results from inference

OFA is evaluated empirically with all OFA variants: tiny, medium, base, large, and huge on zero-shot inference for Image Classification and Visual Question Answering. Images are taken from the pretraining sample dataset and default generator params from the Huggingface repo:

```
generator = sequence_generator.SequenceGenerator(
                    tokenizer=tokenizer,
                    beam_size=3,
                    max_len_b=10, 
                    min_len=0,
                    no_repeat_ngram_size=3,
                    temperature=0.5,
                )
```

For Image Captioning we follow the instruction from the prompt-tuning enhancement for pretraining as highlighted in the ["Prompt Tuning for Generative Multimodal Pretrained Models" paper](https://arxiv.org/pdf/2208.02532) and we use as prompt the following: `[Image] What does the image describe?` and for VQA we are not constrained: `[Image] (Question)`: 

## Image Captioning

<img src="examples/vision_language_examples/teacher.jpg" alt="Image 1" style="height:30%; width:30%;">

<img src="examples/vision_language_examples/goat.jpg" alt="Image 2" style="height:30%; width:30%;">

<img src="examples/vision_language_examples/fisherman.jpg" alt="Image 3" style="height:30%; width:30%;">

|  OFA   | Image 1 | Image 2 | Image 3 |
|--------|---------|---------|---------|
| Tiny   |   a woman wearing a white shirt and blue jeans      |    the dog is black and white     |    the water is calm     |
| Medium |   students in a classroom listening to a presentation      |   a goat in the grass      |    the bridge is now under water     |
| Base   |   a teacher stands in front of her class      |   a black and white goat grazes in a field      |   a man fishing on the rocks at the beach      |
| Large  |   person teaches a class of students      |    a black and white goat grazing on grass     |   a fisherman tries his luck on the beach        |
| Huge   |   students in a classroom with a teacher      |  a goat with a white spot on its back       |    a view of the sea wall at camber sands       |


with bigger temperature - `2.0`, return sequence max length - `20` and no. of beams - `4`:

```
gen = model.generate(inputs, patch_images=patch_img, num_beams=4, no_repeat_ngram_size=3,
                     temperature=2.0, max_length=20, min_length=0, num_return_sequences=1,
                     output_scores=True, output_attentions=True, output_hidden_states=True,
                     return_dict_in_generate=True, use_cache=False)
```


|  OFA   | Image 1 | Image 2 | Image 3 |
|--------|---------|---------|---------|
| Tiny   |   a woman wearing a white shirt and blue jeans      |   the dog is black and white      |   the water is calm      |
| Medium |   students in a classroom listening to a presentation      |   goat in the grass photo #      |   this is a picture of a man fishing from a pier in the water      |
| Base   |   a teacher stands in front of her class      |   a black and white goat grazes in a field       |   a man fishing on the rocks at the mouth of the yarra river       |
| Large  |   person teaches a class of students      |     a black and white baby goat eating grass     |    a fisherman tries his luck on the beach in front of the tappan zee bridge     |
| Huge   |   students in a classroom with a teacher      |  a goat with a white spot on its back       |    people fishing on the shingle beach at weston super mare, north somerset     |

HuggingFace also supports the OFA large variant fine-tuned for Image Captioning: 

| OFA Large Caption (finetune) | Image 1 Caption | Image 2 Caption | Image 3 Caption |
|-----------|----------------|----------------|----------------|
| alt generator | a woman standing in a classroom with a remote in her hand | a black and white goat standing on top of a lush green field | a man standing on a beach holding a fishing pole |
| default generator | a woman standing in a classroom with a remote | a black and white goat grazing in a field | a man standing on a beach holding a fishing pole |

## VQA

For VQA we evaluate qualitatively and empirically zero-shot transfer with examples from multiple benchmarks:` VQA v2, VizWiz, GQA, TextVQA, OK-VQA, A-OK-VQA and Visual Commonsense Reasoning with VCR`. We use the default question as coupeled for each image-text (question) pair:

### VQA v2

<img src="examples/vqa/vqa_v2_1.png" alt="Image 1" style="height:30%; width:30%;">

<img src="examples/vqa/vqa_v2_2.png" alt="Image 2" style="height:30%; width:30%;">

|  Models   | Image 1  | Image 2  |
|--------|-----------------|-----------------|
| Question | **Where is the image on the left taken?** | **Is the woman on the left pouring wine?** | 
| OFA Tiny   |   two umbrellas   |   response 2    |
| OFA Medium |   no    |   response 4    |
| OFA Base   |   in the rain   |   yes   |
| OFA Large  |   a photo    |   response 8    |
| OFA Huge   |   no    |   yes  |
| LLaVa 34b v1.6 |  The image on the left appears to be taken at a beach, as indicated by the sandy ground, the presence of a beach umbrella, and the casual setting with a backpack and a bottle, which are typical items one might bring to a beach. The umbrella is stuck into the sand, and the overall environment suggests a leisurely day at the seaside  |    Yes, the woman on the left is pouring wine | 
| GPT-4o | The image on the left is taken at a beach. This is indicated by the context clues in the image, such as the sand, the beach umbrella, and the other beach-related items | Yes, the woman on the left is pouring wine. This is indicated by the visible action of her holding a wine bottle and pouring its contents into a glass |  
### VizWiz

<img src="examples/vqa/viz_wiz_1.png" alt="Image 1" style="height:30%; width:30%;">

<img src="examples/vqa/viz_wiz_4.png" alt="Image 2" style="height:30%; width:30%;">

<img src="examples/vqa/viz_wiz_3.png" alt="Image 2" style="height:30%; width:30%;">


|  OFA   | Image 1  | Image 2  | Image 3|  
|--------|-----------------|-----------------|-------------|
| Question | **What currency is this?** | **What kind of cleaner is this?** | **What is in the box?**|
| Target | 10 euro | granite daily cleaner | rice |
| Base   |   euro   |   the one on the right is the one with the food on it   | food | 
| Large  |   dollar    |   dish washer/drying/soap    | oats | 
| Huge   |   ten dollars    |   this is a kitchen cleaner that is not a dishwasher  | cereal | 
| LLavA 34b v1.6 | The currency in the image is the Euro. It's a 10 Euro banknote | This is a bottle of "Hope's Perfect Granite Daily Cleaner." It is a product designed specifically for cleaning and maintaining granite surfaces. The label indicates that it is a "100% Streak Free" cleaner, suggesting that it is formulated to leave no streaks or residue on the granite. The price tag on the bottle shows it was sold for $5.99 | The box contains a product that appears to be a rice mix, specifically a "Rice Range" product. It seems to be a meal kit or a mix that can be prepared with additional ingredients to make a meal. The box includes nutrition facts, serving size information, and cooking instructions. The product is designed to be combined with water and other ingredients to create a dish, possibly a pilaf or a similar rice-based meal. The specific flavor or type of rice is not clearly visible in the image provided |
| GPT-4o | The currency shown in the image is a 10 Euro note | This is a granite cleaner. The label on the bottle indicates it is “Hope’s Perfect Granite Daily Cleaner.” | The box contains rice and pasta mix, as indicated by the directions on the back for preparing a dish using rice-pasta mix, margarine or butter, and water. The specific recipe for “Chicken Pilaf” further suggests that the contents are designed for making rice and pasta-based dishes |

### TextVQA

<img src="examples/vqa/text_vqa_giants.png" alt="Image 1" style="height:30%; width:30%;">

<img src="examples/vqa/text_vqa_watch.png" alt="Image 2" style="height:30%; width:30%;">

|  OFA   | Image 1  | Image 2  |
|--------|-----------------|-----------------|
| Question | **What city are the giants from?** | **What number is the small hand on?** | 
| Base   |   new york   |   12   |
| Large  |   new york   |   7    |
| Huge   |   new york    |   7  |
| LLavA 34b v1.6 |  The Giants are from San Francisco | The small hand on the watch is pointing to the number 12 |  
| GPT-4o | The Giants are from San Francisco. This is indicated at the top of the image where it says “San Francisco Giants” | The small hand on the watch is pointing to the number 2 |  


### GQA

<img src="examples/vqa/gqa_1.png" alt="Image 1" style="height:30%; width:30%">

<img src="examples/vqa/gqa_2.png" alt="Image 2" style="height:30%; width:30%">

|  OFA   | Image 1  | Image 2  |
|--------|-----------------|-----------------|
| Question 1 | **Is that a giraffe or an elephant?** | **Is the person's hair brown and long?** | 
| Base   |   giraffe   |   yes   |
| Large  |   giraffe  |   yes   |
| Huge   |   giraffe    |   yes  |
| LLaVa 34b v1.6 |  That is a giraffe. You can tell by its long neck, tall stature, and the distinctive pattern of spots on its body | Yes, the person in the image has long brown hair | 
| GPT-4o  | The animal in the image is a giraffe | Yes, the person in the image has long brown hair | 
| Question 2 | **Who is feeding the giraffe behind the man?** | **What appliance is to the left of the man?** | 
| Base   |  a drink  |   refrigerator   |
| Large  |   woman  |   refrigerator   |
| Huge   |   No    |   refrigerator  |
| LLaVa 34b v1.6 | A woman is feeding the giraffe behind the man. She is holding out her hand with food for the giraffe to eat | There is a refrigerator to the left of the man in the image | 
| GPT-4o  | The person feeding the giraffe behind the man is a woman wearing a gray jacket | The appliance to the left of the man in the image is a refrigerator | 
| Question 3 | **Is there any fence near the animal behind the man?** | **What is the person in front of the refrigerator wearing?** | 
| Base   |   Yes   |   suit   |
| Large  |   Yes  |   suit    |
| Huge   |   Yes    |   suit  |
| LLaVa 34b v1.6 | Yes, there is a fence near the giraffe. The woman is standing behind the fence while feeding the giraffe | The person is wearing a suit and tie |
| GPT-4o  | Yes, there is a fence near the giraffe behind the man. The fence is visible and appears to be made of wood, forming part of the enclosure where the giraffe is located | The person in front of the refrigerator is wearing a black suit with a green tie | 
| Question 4 | **On which side of the image is the man?** | **What is hanging on the wall?** | 
| Target | | picture |
| Base   |   Yes   |   paper   |
| Large  |   Yes  |   pictures      |
| Huge   |   Yes    |   a poster  |
| LLaVa 34b v1.6 | The man is on the right side of the image | There is a paper towel dispenser hanging on the wall | 
| GPT-4o  | The man is on the right side of the image | There is a thermostat hanging on the wall to the left of the man | 

## Knowledge VQA

### OK-VQA

<img src="examples/vqa/kvqa_hydrant.png" alt="Image 1" style="height:30%; width:30%;">

<img src="examples/vqa/kvqa_darthvader.png" alt="Image 1" style="height:30%; width:30%;">

<img src="examples/vqa/kvqa_noah.png" alt="Image 1" style="height:30%; width:30%;">

|  OFA   | Image 1  | Image 2  | Image 3 |
|--------|-----------------|-----------------|---------|
| Question | **What sort of vehicle uses this item?** | **What movie is this character from?** | **What is the name of the man who built the arc in the bible due to this natural catastrophe?** |
| Base   |   fire hydrant  |  batman   | no | 
| Large  |   fire truck   |   star wars   | jesus |
| Huge   |   fire truck   |   star wars  | jesus  | 
| LLaVa 34b v1.6 | The item in the image is a fire hydrant, which is a connection point for firefighters to access water in case of a fire. Fire hydrants are typically connected to a city's water supply system and are used by fire trucks to pump water to extinguish fires | The character in the image is Darth Vader, a central figure from the "Star Wars" franchise. He is known for his distinctive black armor and helmet, as well as his iconic deep, mechanically augmented voice. Darth Vader first appeared in the original "Star Wars" film, released in 1977, and has been a significant character in the subsequent movies, television series, and other media related to the franchise | The man who built the ark in the Bible, as a response to a natural catastrophe, is Noah. The story of Noah and the ark is found in the Book of Genesis in the Bible | 
| GPT-4o | The item in the image is a fire hydrant, which is primarily used by fire trucks. Fire trucks connect hoses to fire hydrants to access a water supply for firefighting | The character in the image is Darth Vader from the movie “Star Wars" | The man who built the ark in the Bible due to a natural catastrophe (a great flood) is Noah | 

Image 3, Alt Question: What natural catastrophe is this?
Re: OFA large & huge: **flood**

## Visual Commonsense Reasoning

<img src="examples/vcr/vcr_3880.jpg" alt="Image 1" style="height:50%; width:50%;">

<img src="examples/vcr/vcr_6310.png" alt="Image 1" style="height:50%; width:50%;">

<img src="examples/vcr/vcr_1.png" alt="Image 1" style="height:50%; width:50%;">

<img src="examples/vcr/vcr_2.png" alt="Image 1" style="height:50%; width:50%;">

For VCR because the answers are more complicated than before, we need to set  the ` min_length` to a higher value - 10, otherwise we get only no/yes.

|  OFA   | Image 1  | Image 2  | Image 3 | Image 4 |
|--------|-----------------|-----------------|---------|-------|
| Question | **Why is [person1] pointing a gun at [person2]?**  | **What would [person1] do if she caught [person2] and [person3] whispering?** | **How did [person2] get the money that’s in front of her?** | **Why is [person4] pointing at [person1]?** | 
| Target | [person1] and [person3] are robbing the bank and [person2] is the bank manager | She would switch their seats | [person2] earned this money playing music | He is telling [person3] that [person1] ordered the pancakes |
| Base   |  [person1] pointing a gun at [person2]  |  blow her hair and then she will be able to hear it/blow her head with her mouth open and tongue out   | [person1] get the money that’s in front of her | [person4] pointing at [person4]  |
| Large  |  to kill the person in the first case of the series/to kill the person in charge of the gun    |  kiss the other woman's lips and mouth while talking to her/punch the man in the face with the stick    | postman [person2 ] got the money out of the box | to see the person eating the food in front of him/he is going to eat the food in front of him | 
| Huge   |  nobody's right hand is pointing at the person in the middle/not possible, because it is not possible to seeVnot possible, because it is not possible to see    |  do you know what she is saying to the other person/  | no indication of who it was, but it was a woman  | the right hand of the person in the middle of the picture | 
| LLaVa 34b v1.6 | The character labeled "[person1]" is pointing a gun at the character labeled "[person2]". However, without additional context from the film or show, it's impossible to determine the exact reason why "[person1]" is pointing a gun at "[person2]". This could be part of a plot, a tense moment in a narrative, or a scene that requires the characters to engage in a conflict. The context of the scene, the characters' motivations, and the overall storyline would provide the necessary information to understand the reason behind this action | The scene shows a classroom setting with a teacher at the front and students in the background.If we were to speculate on what "person1" might do if she caught "person2" and "person3" whispering, it would depend on the context of the scene and the character's personality. Generally, a teacher might ask the students to stop whispering and pay attention to the lesson, or they might ask the students to step outside the classroom for a private conversation to discuss the issue. The teacher's response could range from a gentle reminder to a more stern reprimand, depending on the severity of the disruption and the teacher's teaching style | The context of the scene is not clear from the image alone, so it's not possible to determine how she obtained the money. The character's actions and the plot of the story would provide the necessary information to understand the circumstances surrounding the money. If you have more context or details about the scene, I might be able to offer a more informed interpretation | In the context of the image, it seems that the character labeled as [person4] is pointing at the character labeled as [person1]. However, without additional context from the movie or show, it's impossible to determine the exact reason why [person4] is pointing at [person1]. It could be a part of a conversation, an accusation, a gesture of recognition, or any number of other interactions that are typical in social situations |  
| GPT-4o | To determine why [person1] is pointing a gun at [person2], we need context that the image alone doesn’t provide. However, based on typical narrative scenarios, [person1] could be pointing a gun at [person2] for reasons such as: 1. Conflict: [person1] might see [person2] as a threat or an adversary 2.Interrogation: [person1] might be forcing [person2] to information. 3. Protection: [person1] could be trying to protect someone or something from [person2]. | If [person1] caught [person2] and [person5] whispering, she might likely react by addressing them directly, asking them to stop, or inquiring about what they are discussing. Given the classroom setting and her authoritative posture, it is probable she would take action to maintain order and ensure that the lesson or activity continues without disruption | Based on the context of the image, it is likely that person2 obtained the money in front of her through busking or performing in public. The open instrument case with coins and the tambourine suggest that she and the others might be street performers or musicians who receive donations from passersby for their performances | Based on the context of the image, [person4] (the waiter or waitress) might be pointing at [person1] (the customer) to indicate that the food or drink being served is for them, or they might be confirming an order or directing their attention. This kind of interaction is typical in dining settings where servers ensure that the correct items are delivered to the right customers and address their needs directly |   

Image 2, Alt Question: What class is person1 teaching?
Re: OFA large & huge: english, OFA base: yes/i don't know what i am talking about