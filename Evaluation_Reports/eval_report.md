# Evaluation report

- checkpoint: `/root/.cache/huggingface/hub/models--Hrishikesh1903--gpt124m-ckpt/snapshots/6b05b497b705f72a16d87418e1ad0f3411808bb8/model_final.pt`
- parameters: 124.5M, device: cuda

## Held-out perplexity

| Model | Loss (nats/token) | Perplexity | Bits/token | Bits/byte | Tokens |
|---|---|---|---|---|---|
| This model | 4.2420 | 69.54 | 6.120 | 1.328 | 9,999,360 |
| OpenAI GPT-2 124M (reference) | 3.2770 | 26.50 | 4.728 | 1.026 | 9,999,360 |

## Generation quality

| Decoding | Samples | distinct-1 | distinct-2 | distinct-3 | Repeated 4-grams | Avg new tokens |
|---|---|---|---|---|---|---|
| greedy | 10 | 0.112 | 0.130 | 0.142 | 84.8% | 100 |
| sampled | 30 | 0.487 | 0.752 | 0.845 | 10.2% | 97 |
| sampled+rep_penalty | 30 | 0.687 | 0.908 | 0.961 | 1.9% | 93 |

Higher distinct-n means more varied text; a high repeated-4-gram share means looping.

## Samples (first sample per prompt and decoding setting)

**[greedy]** `Photosynthesis is the process by which plants`

> Photosynthesis is the process by which plants are grown in the soil.
> The plant is a plant that is grown in the soil. It is a plant that is grown in the soil. It is a plant that is grown in the soil. It is grown in the soil and is grown in the soil.
> The plant is grown in the soil. It is grown in the soil. It is grown in the soil. It is grown in the soil. It is grown in the soil. It is grown in the soil. It is

**[greedy]** `The French Revolution began in 1789 when`

> The French Revolution began in 1789 when the French Revolution was established in 1789.
> The French Revolution was the first to be known as the French Revolution. The French Revolution was the first to be known as the French Revolution.
> The French Revolution was the first to be known as the French Revolution.
> The French Revolution was the first to be known as the French Revolution.
> The French Revolution was the first to be known as the French Revolution.
> The French Revolution was the first to be known as the French Revolution.
> The

**[greedy]** `In this lesson, students will learn how to`

> In this lesson, students will learn how to use the lesson plans to use the lesson plans.
> - Students will learn how to use the lesson plans to use the lesson plans.
> - Students will learn how to use the lesson plans to use the lesson plans.
> - Students will learn how to use the lesson plans to use the lesson plans.
> - Students will learn how to use the lesson plans to use the lesson plans.
> - Students will learn how to use the lesson plans to use the lesson plans.
> - Students will learn

**[greedy]** `The human heart is divided into four chambers:`

> The human heart is divided into four chambers: the heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the heart. The heart is the

**[greedy]** `One of the main causes of climate change is`

> One of the main causes of climate change is the most common cause of climate change.
> The climate change is caused by climate change.
> The climate change is caused by climate change.
> The climate change is caused by climate change.
> The climate change is caused by climate change.
> The climate change is caused by climate change.
> The climate change is caused by climate change.
> The climate change is caused by climate change.
> The climate change is caused by climate change.
> The climate change is caused by climate change.
> The

**[greedy]** `To calculate the area of a triangle, you need to`

> To calculate the area of a triangle, you need to calculate the area of the triangle.
> The graph below shows the area of the triangle.
> The graph below shows the area of the triangle.
> The graph below shows the area of the triangle.
> The graph below shows the area of the triangle.
> The graph below shows the area of the triangle.
> The graph below shows the area of the triangle.
> The graph below shows the area of the triangle.
> The graph below shows the area of the triangle.
> The graph below shows

**[greedy]** `The Industrial Revolution had a major impact on`

> The Industrial Revolution had a major impact on the economy. The economy was a major factor in the economy. The economy was a major factor in the economy. The economy was a major factor in the economy. The economy was a major factor in the economy. The economy was a major factor in the economy. The economy was a major factor in the economy. The economy was a major factor in the economy. The economy was a major factor in the economy. The economy was a major factor in the economy. The economy was a major factor in

**[greedy]** `Water is essential for life because`

> Water is essential for life because it is a natural resource for the environment.
> - It is a natural resource for the environment.
> - It is a natural resource for the environment.
> - It is a natural resource for the environment.
> - It is a natural resource for the environment.
> - It is a natural resource for the environment.
> - It is a natural resource for the environment.
> - It is a natural resource for the environment.
> - It is a natural resource for the environment.
> - It

**[greedy]** `The Pythagorean theorem states that`

> The Pythagorean theorem states that the universe is the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the same as the

**[greedy]** `A healthy diet should include a variety of`

> A healthy diet should include a variety of foods, such as fruits, vegetables, and vegetables.
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy diet
> - Eat a healthy

**[sampled]** `Photosynthesis is the process by which plants`

> Photosynthesis is the process by which plants have been exposed to this process.
> The plants used for sunlight in the soil to break down, and we now have an array of microbes that have come into the soil and the plants we produce. It has been shown that the plants produced are the ones that we now have been able to release from the surface and the plants that we do not need to do it.
> The process begins with a lot of water that is stored in water, so we can see that when plants are harvested, plants

**[sampled]** `The French Revolution began in 1789 when`

> The French Revolution began in 1789 when British troops took the same territory to their own territories and they moved south to the French territory. On the following day, British troops crossed the island of San Jose and fled the island of Santa Maria de Taurua.
> The United States Army of the United States was the first to decide to hold the country's territory as a treaty that allowed them to have their own territories. In March, 1871, the U.S. Supreme Court of Rights was elected to the United States and was founded

**[sampled]** `In this lesson, students will learn how to`

> In this lesson, students will learn how to read and write, which is a free, fun lesson. I can be a little longer a year.
> “This is the great book of fun! I will always have a few videos and then you can use a lot of my own.”
> You can also learn how to read and remember the book.
> “We’re not going to give a quick, creative way, but they’re good things. There are a lot of people who need to think

**[sampled]** `The human heart is divided into four chambers:`

> The human heart is divided into four chambers:
> - The human brain is divided into two sections:
> - The human brain is divided into four sections:
> - The human body is divided into five sections:
> - The human body is divided into four sections:
> - The human body is divided into four sections:
> - The human body is divided into three sections:
> - The human body is divided into three sections:
> - The human body is divided into four sections:
> - The human body is divided into two groups:
> 

**[sampled]** `One of the main causes of climate change is`

> One of the main causes of climate change is due to climate change and climate change.
> A few examples of environmental stressors, particularly on the climate, are caused by climate change.
> A global climate is a challenge to climate change, or in a global climate.
> For all, the climate is a challenge to climate change.
> Global climate impacts have been defined as a “polar climate change”.
> An international team of experts in the World Health Organization say that the average annual annual annual annual annual annual annual annual annual

**[sampled]** `To calculate the area of a triangle, you need to`

> To calculate the area of a triangle, you need to get the following formula and the following formula. This should include any two terms, so you can go to the left and then get the formula.
> How to use the cube to calculate the region of a triangle?
> Use the matrix to calculate the area of a triangle.
> When you use the cube to calculate the area of a triangle, you need to get the solution. This will help you understand what is in the square.
> A square has a square and square with a circle of triangle

**[sampled]** `The Industrial Revolution had a major impact on`

> The Industrial Revolution had a major impact on the economy and its role in the economy. This has been linked to the economic problems of the economy and the economy. This is a critical step toward the growth and development of the economy.
> How many countries have been taking the market with these countries?
> The market economy is a major factor in the future of the economy. The economy is the main factor in the economy. This is the major factor in the economy. It is also the major factor in the economy. The economic growth in the

**[sampled]** `Water is essential for life because`

> Water is essential for life because of the development of this essential oil is essential for health. For these reasons, a number of people are deficient in these oils, and their body fat may be able to convert it to it. Moreover, a number of people are less likely to have a higher level of fat.
> According to the American College of Medicine, the American College of Medicine recommends that the American College of Medicine recommends that people with high blood pressure, especially those who have low blood pressure, should use this treatment, and then

**[sampled]** `The Pythagorean theorem states that`

> The Pythagorean theorem states that it is more likely than the other
> contrastical, more difficult to understand what this theorem explains.
> 1. In this view, it is possible that it is less likely to be better to achieve a higher level of
> sugar-free, low-fat, and low-fat.
> 2. In this view, it is possible that the equation is greater than that of the
> contrastical and non-nutrients in the sense of the equation is better than
> this hypothesis

**[sampled]** `A healthy diet should include a variety of`

> A healthy diet should include a variety of fruits and vegetables that have a high incidence of cancer. The following foods should be balanced or a good diet should be checked by a general rule of thumb, but should be made to avoid foods that have an impact on your health.
> Foods should be kept in a balanced diet and should be set to reduce the risk of cancer. You should be aware that the following foods should be given to your health and the health of your diet.
> The key to the list of foods that have a strong

**[sampled+rep_penalty]** `Photosynthesis is the process by which plants`

> Photosynthesis is the process by which plants have been exposed to this plant.
> The plants of this plant are now being developed in the same area as this plant is very hard. It can be used to grow the leaves and stems of the seeds that are not healthy but also too much, so it will eventually become a problem for many years. These flowers are usually grown in an easy-to-date environment where they are still growing in areas like New Mexico.

**[sampled+rep_penalty]** `The French Revolution began in 1789 when`

> The French Revolution began in 1789 when the British Revolution had to be replaced by a revolution that would soon change.
> “When we were not so bad, the man who was very weak and strong, and he was very close to him and his own power.”
> In 1894, the British Revolution saw the American Revolution as an attempt to overthrow the British Revolution. In 1779, the British Revolution ended with the French Revolution and became part of the European Revolution.
> This was the first war between France and America.

**[sampled+rep_penalty]** `In this lesson, students will learn how to`

> In this lesson, students will learn how to teach the lessons they’ve learned. This is an activity that I have not found that there are many questions about this topic.
> I think we are learning about math and science because it helps teachers develop their own understanding of the mathematics in a way that allows them to understand the concepts of the world. It also encourages students to recognize and understand what they know about.
> We believe that the students can learn the skills that they learn to learn as well as how they know.
> It provides

**[sampled+rep_penalty]** `The human heart is divided into four chambers:`

> The human heart is divided into four chambers:
> - The body can be divided into two sections. Each and every part of the heart is given to each other. A large portion of the heart has a total number of parts, but there are two different types of organs. This is why it is important to know how to use this information in order to get the same results. This article is based on a paper published by David W. Weaver.

**[sampled+rep_penalty]** `One of the main causes of climate change is`

> One of the main causes of climate change is, in part due to the fact that climate change impacts on Earth.
> The problem with climate change is that they are more likely to happen when we want to take a closer look at our climate change. This is because we have many of the most important environmental and economic benefits of climate change.
> In an attempt to develop a climate change situation, climate change can be linked to the weather conditions of a global climate change. For example, one of the most significant factors to cause climate change are not

**[sampled+rep_penalty]** `To calculate the area of a triangle, you need to`

> To calculate the area of a triangle, you need to calculate the area of an angle.
> The square circle of an area is divided into two categories:
> - The width of a rectangle has four parts and two sides.
> - The size of an area in which the area is square.
> - The height of the area in which the area is square.
> - The length of the area is square.
> - The height of the area in which a triangle is square.
> - The length of the area in which the area of an area

**[sampled+rep_penalty]** `The Industrial Revolution had a major impact on`

> The Industrial Revolution had a major impact on the economy and was one of the most important industries in Europe. By the time, people were faced with the largest economic development projects in Germany and the Netherlands. The majority of the world’s economy has been affected by the increased risk of the economy and the number of people living in the US.
> As the economic growth rate increases, it is estimated that about 90% of the population will be considered to be the “carbon inequality” of the world’s population. According

**[sampled+rep_penalty]** `Water is essential for life because`

> Water is essential for life because of the development of this plant.
> The following questions will be found in the list below:
> - The growing number of plants and seeds in the US is about 6% (1).
> - All of these plants are not in the United States, but also in the United States.
> - The annual number of plants in the USA is around 5% and the total number of plants is approximately 4.6 % (3.2%) of the plants is about 9.7%.
> -

**[sampled+rep_penalty]** `The Pythagorean theorem states that`

> The Pythagorean theorem states that it is the most fundamental to the existence of any universe, in order to be used.
> If you are interested in an experiment or study on this question, please contact us at firstname.lastname@example.org

**[sampled+rep_penalty]** `A healthy diet should include a variety of`

> A healthy diet should include a variety of different foods, including nuts and vegetables.
> What’s the difference between fruits and vegetables?
> - Try to find out how your body can control and manage nutrients from the food chain to try:
> - Eat healthy foods that are low in calories
> - Eat whole grains and vegetables
> - Eat healthier
> - Eat more dairy products
> - Avoid eating less meat
> - Avoid fish oil
> - Drink plenty of water
> - Find foods rich in antioxidants
> - Use vitamins A, B6
