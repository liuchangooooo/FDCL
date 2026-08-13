# Towards Adaptive Environment Generation for Training Embodied Agents

Teresa Yeo \( {}^{1 * } \) , Dulaj Weerakoon \( {}^{1 * } \) , Dulanga Weerakoon \( {}^{1} \) , Archan Misra \( {}^{2} \)

\( {}^{1} \) Singapore-MIT Alliance for Research and Technology Centre

\( {}^{2} \) Singapore Management University

\{teresa.yeo, dulaj.weerakoon, dulanga.weerakoon\}@smart.mit.edu, archanm@smu.edu.sg

## Abstract

Embodied agents struggle to generalize to new environments, even when those environments share similar underlying structures to their training settings. Most current approaches to generating these training environments follow an open-loop paradigm, without considering the agent's current performance. While procedural generation methods can produce diverse scenes, diversity without feedback from the agent is inefficient. The generated environments may be trivially easy, providing limited learning signal. To address this, we present a proof-of-concept for closed-loop environment generation that adapts difficulty to the agent's current capabilities. Our system employs a controllable environment representation, extracts fine-grained performance feedback beyond binary success or failure, and implements a closed-loop adaptation mechanism that translates this feedback into environment modifications. This feedback-driven approach generates training environments that more challenging in the ways the agent needs to improve, enabling more efficient learning and better generalization to novel settings.

## Introduction

Embodied agents often struggle to generalize to new environments, even when those environments share similar underlying structures to their training settings. Consider a navigation agent trained in a simple apartment layout. It may successfully navigate from a starting position to a target location, such as a refrigerator, in its training environment. However, when deployed in a visually similar apartment where furniture and objects have been rearranged, the same agent may fail to reach its goal (Fig. 1). Thus, without sufficient diversity in the training data, agents can fail to develop robust navigation strategies that transfer to new settings.

Embodied agents, such as home assistance robots or industrial manipulators, face a data scarcity challenge. Unlike disembodied agents like ChatGPT (OpenAI 2022), and Gemini (Anil et al. 2025) that are trained on massive internet-scale datasets, embodied agents tend to be trained using data generated in simulated physical environments. Creating these simulations can be expensive and time-consuming, making training data more scarce than for their disembodied counterparts. This scarcity represents a bottleneck in developing robust embodied agents.

![0_937_623_705_320_0.jpg](images/0_937_623_705_320_0.jpg)

Figure 1: Embodied navigation performance is sensitive to object perturbations. Top-down view of agent trajectories (yellow to orange path) for an object navigation task with the fridge as the target object. In the training environment (left), the agent successfully navigates to the target, while in the test environment (right) when the objects and furniture has been rearranged, the agent fails to reach the same target despite starting from the same position. Orange cross indicates target location. Our method analyzes such failure trajectories to generate difficulty-aware environments by programmatically modifying ProcTHOR scene structures, creating meaning and challenging scenarios that can be used to train and improve the agent. Best viewed on screen, zoomed in.

Most existing approaches to generating training environments follow an open-loop paradigm, i.e., they generate environments without taking into account the agent's current performance. Hand-crafted environments can be of high quality but require domain expertise and time to create. Alternatively, procedural generation methods can produce diverse scenes by randomly sampling from templates and object databases. For example, by scanning a real room and generating variations (Deitke et al. 2023). While random generation increases diversity, it does not necessarily result in environments that are useful for training e.g., some generated environments may be trivially easy, providing no meaningful learning signal. Thus, diversity without feedback is inefficient: we need to generate environments that are appropriately challenging for the given agent we are training.

This paper proposes a proof-of-concept for closed-loop environment generation where the difficulty of generated environments is adapted to a given agent's current capabilities. Our system includes three key components. First, we employ a controllable environment representation that can be modified efficiently and programmatically. Second, we extract fine-grained feedback signals from the agent's performance, going beyond binary success or failure. Third, we implement a closed-loop adaptation mechanism that translates this feedback into environment modifications. These components create an adaptive curriculum where training environments grow progressively more challenging in precisely the ways the agent needs to improve, enabling more efficient learning and better generalization to novel settings.

---

*These authors contributed equally.

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

---

## Related Work

Environment Generation Approaches. Environment simulators can be broadly categorized by their generation approach. One category uses scans from real-world environments, e.g., Habitat with the Matterport3D dataset (HM3D) (Ramakrishnan et al. 2021), Replica (Straub et al. 2019), ScanNet (Dai et al. 2017), and iGibson (Li et al. 2021), providing photo-realistic reconstructions of actual indoor spaces, from homes to commercial buildings. A second category employs procedural generation techniques to create synthetic environments, as demonstrated by AI2-Thor (Kolve et al. 2017), ProcThor (Deitke et al. 2022), ThreeDWorld (Gan et al. 2020), Infinigen (Raistrick et al. 2024). Recently, environments can also be generated from natural language descriptions with systems like Genie (Bruce et al. 2024) and Genesis (Zhou et al. 2024), allowing for more flexible and user-directed scene creation.

Each approach presents distinct trade-offs. Real-world scans yield highly realistic environments but lack controllability and are less suited for systematic adaptation. Procedurally generated environments offer high controllability at the cost of visual realism. While recent language-based systems like Genie produce realistic scenes, prompting does not allow for the fine-grained control needed for precise scene manipulation. Thus, we selected ProcThor for environment generation due to its balance of controllability and systematic scene composition capabilities.

Open-loop Environment Generation. Open-loop generation methods create environments without considering agent performance or task outcomes. These approaches can be further divided into hand-crafted and sampling methods. Hand-crafted approaches like Holodeck (Yang et al. 2024), LayoutVLM (Sun et al. 2025), and AnyHome (Fu et al. 2024) design individual environments through hand-crafted specifications. Sampling approaches like RoboGen (Wang et al. 2023) and Phone2Proc (Deitke et al. 2023) generate large collections of diverse environments through random sampling procedures. While these open-loop methods can produce diverse training data, they do not leverage feedback from agent's performance to guide environment generation. In contrast, our work employs a closed-loop approach that adapts environment generation based on agent performance, potentially improving data efficiency by focusing on more informative training scenarios.

Closed-loop environment generation Closed-loop environment generation automatically creates training environments that adapt to an agent's learning progress. Kanitschei-der et al. (2021) established learning progress-measured as changes in task success probability—as a reliable signal for curriculum construction in complex domains like Minecraft, demonstrating that dynamically selecting environments based on what the agent can learn next substantially improves training efficiency. Building on adaptive curriculum principles, ADD (Chung et al. 2024) uses diffusion models guided by agent regret to generate diverse adversarial environments that are challenging yet learnable, leveraging the representational power of diffusion models to overcome limitations of parameterized environment spaces.

Recent work uses foundation models for generating environments: OMNI-EPIC (Faldor et al. 2024) uses LLMs to generate both environment and reward function code, enabling open-ended creation of any simulatable task while maintaining interestingness and learnability. Similarly, Eu-rekaverse (Liang et al. 2024) applies LLM-based code generation to robotic learning through agent-environment co-evolution, automatically designing terrain curricula for quadrupedal parkour. Our work also makes use of these foundation models for closed-loop environment generation.

## Method

We now describe our framework for adaptively generating training environments. See Fig. 2 for an illustration.

Problem formulation. Let \( e \in  \mathcal{E} \) denote an environment configuration, represented by a structured scene graph or parameterized layout. An agent with policy \( \pi \left( {a \mid  s}\right) \) performs embodied tasks such as object navigation, or manipulation. When deployed on environment \( e \) , the agent produces a trajectory \( {\tau }^{e} \in  \mathcal{T} \) , which can be represented as a sequence of states and actions or as a top-down view (see Fig. 1).

We define a function \( F : \mathcal{T} \rightarrow  \mathcal{A} \) (e.g., an LLM like GPT- 5-mini) that takes the top-down view of trajectory \( {\tau }^{e} \) and returns an analysis \( a \in  \mathcal{A} \) describing if the agent succeeded or failed, identifying concerns, and providing high-level suggestions for environment modification. Next, we define an environment generator \( G : \mathcal{E} \times  \mathcal{A} \rightarrow  \mathcal{E} \) (also an LLM), which at iteration \( t \) , takes the current environment configuration \( {e}_{t} \) and the trajectory analysis \( {a}_{t} = F\left( {\tau }^{{e}_{t}}\right) \) as input, and outputs a new environment configuration \( {e}_{t + 1} = G\left( {{e}_{t},{a}_{t}}\right) \) .

The goal of \( G \) is to produce environments that are both sufficiently challenging and diverse for the agent. Formally, \( G \) aims to maximize the following objective:

\[
\mathcal{J}\left( G\right)  = {\mathbb{E}}_{t}\left\lbrack  {{\Delta R}\left( {{\pi }_{t}, G\left( {{e}_{t}, F\left( {\tau }^{{e}_{t}}\right) }\right) }\right) }\right\rbrack
\]

where \( {\Delta R} \) measures the improvement in agent performance on held-out environments. This formulation establishes a feedback loop between the generator and the agent, allowing for procedural and adaptive environment generation.

Structured representation of the environment. The environment configuration \( e \) can be represented as a structured graph capturing the composition of the scene. One such representation is \( e = \left( {O, A, R}\right) \) , where \( O = \left\{  {o}_{i}\right\} \) denotes the set of objects, \( A\left( {o}_{i}\right) \) the attributes of each object (e.g., position, rotation, scale, material), and \( R\left( {{o}_{i},{o}_{j}}\right) \) the pairwise spatial or functional relations (e.g., on, next-to, inside). This representation can be instantiated in simulators such as AI2-THOR, where an environment is defined by a configuration file specifying object types and their parameters, forming a directed scene graph.

![2_210_150_1384_599_0.jpg](images/2_210_150_1384_599_0.jpg)

Figure 2: Overview of the proposed adaptive environment generation framework. Starting from an original environment \( {e}_{t} \) , an agent with policy \( {\pi }_{t} \) is deployed to perform embodied tasks (e.g., object navigation), producing a top-down trajectory visualization \( {\tau }^{{e}_{t}} \) . An analysis model \( F \) examines this trajectory to identify success/failure status, intermediate concerns (e.g., unsafe navigation margins), and high-level suggestions for curriculum design. The generator \( G \) then translates this analysis \( {a}_{t} \) into concrete scene graph modifications (e.g., moving or rotating objects) to create a perturbed environment \( {e}_{t + 1} \) . The agent is retrained on this new environment with policy \( {\pi }_{t + 1} \) , completing the feedback loop between adaptive environment generation and agent learning.

Such a structured representation enables \( G \) to perform modifications like adding, removing, or perturbing objects. Furthermore, it also enables a separate validation module to check physical consistency and task feasibility (e.g., ensuring that objects do not overlap and tasks remain solvable). This is discussed at the end of the section.

Fine-grained trajectory analysis as adaptation signal. To generate feedback signals for environment adaptation, we employ \( F \) to analyze the agent’s trajectory i.e., \( F \) is given an image like the one on Fig. 1 (left) to analyze. Unlike traditional reinforcement learning approaches that rely on binary task completion signals, our method extracts structured, intermediate-state feedback from the trajectory. Given the agent's trajectory and task description, \( G \) produces:

Task outcome assessment: If the agent succeeded or failed. Intermediate concerns: Behavioral issues observed during execution (e.g., unsafe clearances, inefficient paths).

Suggestions: Abstract suggestions that specify what aspects of the environment should change to address the concerns, but not the how (e.g., object placement locations).

\( F \) maps the trajectory to structured feedback: \{outcome, concerns, suggestions\}. The concerns highlight specific states or transitions where the agent's behavior is suboptimal, even when the task succeeds. The adaptations specify abstract environment modifications suggestions to create training scenarios to address these concerns. Note \( F \) does not propose low-level suggestions on how to create these scenarios i.e., where to move certain objects.

This approach exploits the rich intermediate information available in embodied navigation trajectories, enabling generation of targeted training environments that addresses specific weaknesses of the agent.

Adaptive environment generation. Given \( F \) ’s analysis \( {a}_{t} \) of the agent’s trajectory and the environment configuration \( {e}_{t} \) , the generator \( G \) aims to produce a more challenging environment configuration \( {e}_{t + 1} = G\left( {{e}_{t},{a}_{t}}\right) \) for the agent. Concretely, \( G \) aims to generate an \( {e}_{t + 1} = \; \arg \mathop{\min }\limits_{{{e}^{\prime } \in  \mathcal{E}}}R\left( {{\pi }_{t},{e}^{\prime }}\right) \) subject to validity and task solvability constraints. The agent is then trained in the new environment \( {e}_{t + 1} \) . This results in an updated policy \( {\pi }_{t + 1} \) . This interaction between the agent and the generator results in a curriculum where the generator continually synthesizes harder environments that are adapted to the agent's current skills.

The generator \( G \) can be instantiated in multiple ways depending on the desired level of structure and interpretability. One approach leverages an LLM that conditions on the agent’s trajectory or top-down map of its behavior in \( {e}_{t} \) , and outputs a sequence of discrete editing actions-e.g., selecting an object \( {o}_{i} \in  O \) and modifying its spatial parameters or relations to create a new configuration \( {e}_{t + 1} \) . As LLMs possess spatial reasoning and world-model priors, this formulation enables broad generalization to unseen scenes and tasks, often producing semantically coherent and meaningful modifications without task-specific training. In contrast, \( G \) can also be trained to predict configuration deltas given \( {e}_{t} \) and the reward \( {a}_{t} \) . This approach supports gradient-based optimization, lower latency, and scalability across many agents or tasks, but may require task-specific data and tends to generalize less reliably beyond the training distribution.

Constrained optimization to ensure physical plausibility. The modifications from \( G \) can result in object placements that are not physically plausible e.g., objects intersecting with other objects or walls etc (see Fig. 3). To perform collision-aware placement, we formulate the problem as follows: given an object \( {o}_{i}^{t} \) with attributes \( \mathcal{A}\left( {o}_{i}^{t}\right) \) that include its original position \( {\mathbf{p}}_{i}^{\text{ orig }} \) , and a proposed target position \( {\mathbf{p}}_{i}^{\text{ target }} \) generated by \( G \) , we compute the displacement vector \( \mathbf{d} = {\mathbf{p}}_{i}^{\text{ target }} - {\mathbf{p}}_{i}^{\text{ orig }} \) . We then move \( {o}_{i}^{t} \) along this direction in discrete steps of size \( \delta \) , updating its position component in \( \mathcal{A}\left( {o}_{i}^{t}\right) \) as \( {\mathbf{p}}_{i}\left( k\right)  = {\mathbf{p}}_{i}^{\text{ orig }} + {k\delta }\frac{\mathbf{d}}{\parallel \mathbf{d}\parallel } \) , where \( k \) denotes the iteration step, while keeping other attributes (rotation, scale, material) unchanged.

![3_156_146_704_290_0.jpg](images/3_156_146_704_290_0.jpg)

Figure 3: Collision-aware placements. When the generator \( G \) proposes a new environment configuration that would result in object collisions, we apply a collision-aware adjustment to find a valid placement. In this example, the chair is initially proposed to move 20 units in the x-direction and 5 units in the y-direction, which would cause it to collide with the bed. We compute the displacement vector from the chair's current position to the proposed position and moves the chair incrementally along this direction. At each step, we check for collisions with other objects. This continues until either the proposed position is reached without collision or an obstacle blocks further movement along the path, at which point the chair is placed at the last valid collision-free position.

At each iteration, we perform collision detection between \( {o}_{i}^{t} \) and all other objects in the scene. The adjustment process terminates under two conditions: 1. \( {o}_{i}^{t} \) reaches \( {\mathbf{p}}_{i}^{\text{ target }} \) without collision (i.e., \( {k\delta } \geq  \parallel \mathbf{d}\parallel \) ), or 2. a collision is detected at step \( \mathrm{k} \) , in which case we update \( \mathcal{A} \) with the last valid collision-free position \( {\mathbf{p}}_{i}\left( {k - 1}\right) \) . This approach ensures that the final configuration respects spatial constraints while attempting to satisfy the proposed modification as closely as possible.

## Experiments

Setup. We employ GPT-4.1-mini for both the analysis function \( F \) and the modification function \( G \) . For the navigation agent, we utilize a pre-trained policy \( \pi \) from Spoc \( {}^{1} \) (Ehsani et al. 2024), which has been trained via imitation learning on millions of frames of shortest-path expert trajectories. Given \( F \) ’s trajectory analysis \( a \) of policy \( \pi , G \) modifies the environment one object at a time: it selects an object and proposes a modification (x-y displacement and/or rotation). We use Structured Outputs (via the OpenAI API) to constrain \( G \) ’s response format. Specifically, \( G \) selects an object from the list of movable objects in the scene and output: 1. the selected object's identifier, 2. its x-y displacement values, and 3. a rotation angle. This modification is then applied using collision-aware placement. After each modification, the updated top-down environment is rendered and passed to \( G \) to generate the next modification. After several steps, we get the final modified environment. See the Appendix for the prompts used with \( F \) and \( G \) .

Preliminary results. Fig. 2 shows an example of the perturbed environment \( {e}_{t + 1} \) . This new environment takes into account the analysis from \( F \) by e.g., creating narrower pathways. Furthermore, the perturbed environment is more realistic than randomly perturbing each object. This proof-of-concept demonstrates the feasibility of our approach, thorough comprehensive evaluation including baseline comparisons and agent training on generated environments remains as future work.

Limitations of frontier LLM Spatial Reasoning Capabilities. Despite the promise shown in our proof-of-concept, frontier LLMs have known limitations in spatial reasoning. While \( G \) can propose modifications in terms of coordinates and rotations, it may struggle to accurately visualize the spatial consequences of these changes, e.g., if moving a sofa 2 units along the x-axis creates its intended effect. The top-down visualization provided to the model helps mitigate this, but LLMs can still propose modifications that are plausible but are spatially incoherent. To address this, we experimented with a verification step: after each modification is rendered, we pass the updated visualization back to \( G \) and ask if the change matches its intended effect. If not, \( G \) can revise its modification. This mechanism helps catch spatial misunderstandings. Note that this differs from the collision-aware placement which ensures physically plausible configurations; the limitation here is with \( G \) ’s mapping of high-level intentions (e.g., creating narrower pathways) to the actual modifications needed to achieve them. These limitations suggest that 1. the iterative feedback loop is crucial, allowing \( G \) to observe rendered results and correct, and 2. future work might benefit from hybrid approaches that combine frontier LLMs with modules that have been trained to perform spatial reasoning for object manipulation.

## Conclusion and future work

This work proposes a closed-loop pipeline for generating new environments for training navigation agents. We assume that the environment can be represented as a structured graph (via simulators like ProcThor). Closed-loop adaptation is then performed by using off-the-shelf LLMs to analyze the agent's trajectory on existing environments and proposing modifications to make this environment more challenging for a given navigation agent. We demonstrate the feasibility of this approach through proof-of-concept implementations. While our results are preliminary, it shows the potential of using LLM-driven closed-loop generation for adaptive environment design. Future work will include comprehensive evaluation against baselines and training navigation agents on the generated environments (over multiple iterations of environment generation and training) to validate the effectiveness of our approach.

---

\( {}^{1} \) SigLIP-ViTb-3-CHORESNav-S, downloaded from https:// github.com/allenai/spoc-robot-training

---

## Acknowledgments

This research is supported by the National Research Foundation (NRF), Prime Minister's Office, Singapore under its Campus for Research Excellence and Technological Enterprise (CREATE) programme. The Mens, Manus, and Machina (M3S) is an interdisciplinary research group (IRG) of the Singapore MIT Alliance for Research and Technology (SMART) centre.

## References

Anil, R.; Borgeaud, S.; Alayrac, J.-B.; et al. 2025. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805.

Bruce, J.; Dennis, M.; Edwards, A.; Parker-Holder, J.; Shi, Y.; Hughes, E.; Lai, M.; Mavalankar, A.; Steigerwald, R.; Apps, C.; Aytar, Y.; Bechtle, S.; Behbahani, F.; Chan, S.; Heess, N.; Gonzalez, L.; Osindero, S.; Ozair, S.; Reed, S.; Zhang, J.; Zolna, K.; Clune, J.; de Freitas, N.; Singh, S.; and Rocktäschel, T. 2024. Genie: Generative Interactive Environments. arXiv:2402.15391.

Chung, H.; Lee, J.; Kim, M.; Kim, D.; and Oh, S. 2024. Adversarial environment design via regret-guided diffusion models. Advances in Neural Information Processing Systems, 37: 63715-63746.

Dai, A.; Chang, A. X.; Savva, M.; Halber, M.; Funkhouser, T.; and Nießner, M. 2017. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, 5828-5839.

Deitke, M.; Hendrix, R.; Farhadi, A.; Ehsani, K.; and Kemb-havi, A. 2023. Phone2proc: Bringing robust robots into our chaotic world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9665-9675.

Deitke, M.; VanderBilt, E.; Herrasti, A.; Weihs, L.; Salvador, J.; Ehsani, K.; Han, W.; Kolve, E.; Farhadi, A.; Kembhavi, A.; and Mottaghi, R. 2022. ProcTHOR: Large-Scale Embodied AI Using Procedural Generation. In NeurIPS. Outstanding Paper Award.

Ehsani, K.; Gupta, T.; Hendrix, R.; Salvador, J.; Weihs, L.; Zeng, K.-H.; Singh, K. P.; Kim, Y.; Han, W.; Herrasti, A.; et al. 2024. Spoc: Imitating shortest paths in simulation enables effective navigation and manipulation in the real world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16238-16250.

Faldor, M.; Zhang, J.; Cully, A.; and Clune, J. 2024. Omni-epic: Open-endedness via models of human notions of interestingness with environments programmed in code. arXiv preprint arXiv:2405.15568.

Fu, R.; Wen, Z.; Liu, Z.; and Sridhar, S. 2024. Anyhome: Open-vocabulary generation of structured and textured 3d homes. In European Conference on Computer Vision, 52- 70. Springer.

Gan, C.; Schwartz, J.; Alter, S.; Mrowca, D.; Schrimpf, M.; Traer, J.; De Freitas, J.; Kubilius, J.; Bhandwaldar, A.; Haber, N.; et al. 2020. Threedworld: A platform for interactive multi-modal physical simulation. arXiv preprint arXiv:2007.04954.

Kanitscheider, I.; Huizinga, J.; Farhi, D.; Guss, W. H.; Houghton, B.; Sampedro, R.; Zhokhov, P.; Baker, B.; Ecof-fet, A.; Tang, J.; et al. 2021. Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft. arXiv preprint arXiv:2106.14876.

Kolve, E.; Mottaghi, R.; Han, W.; VanderBilt, E.; Weihs, L.; Herrasti, A.; Deitke, M.; Ehsani, K.; Gordon, D.; Zhu, Y.; et al. 2017. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474.

Li, C.; Xia, F.; Martín-Martín, R.; Lingelbach, M.; Srivas-tava, S.; Shen, B.; Vainio, K.; Gokmen, C.; Dharan, G.; Jain, T.; et al. 2021. igibson 2.0: Object-centric simulation for robot learning of everyday household tasks. arXiv preprint arXiv:2108.03272.

Liang, W.; Wang, S.; Wang, H.-J.; Bastani, O.; Jayaraman, D.; and Ma, Y. J. 2024. Eurekaverse: Environment curriculum generation via large language models. arXiv preprint arXiv:2411.01775.

OpenAI. 2022. Introducing ChatGPT. https://openai.com/ index/chatgpt/. Accessed November 7, 2025.

Raistrick, A.; Mei, L.; Kayan, K.; Yan, D.; Zuo, Y.; Han, B.; Wen, H.; Parakh, M.; Alexandropoulos, S.; Lipson, L.; et al. 2024. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21783-21794.

Ramakrishnan, S. K.; Gokaslan, A.; Wijmans, E.; Maksymets, O.; Clegg, A.; Turner, J.; Undersander, E.; Galuba, W.; Westbury, A.; Chang, A. X.; et al. 2021. Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238.

Straub, J.; Whelan, T.; Ma, L.; Chen, Y.; Wijmans, E.; Green, S.; Engel, J. J.; Mur-Artal, R.; Ren, C.; Verma, S.; et al. 2019. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797.

Sun, F.-Y.; Liu, W.; Gu, S.; Lim, D.; Bhat, G.; Tombari, F.; Li, M.; Haber, N.; and Wu, J. 2025. Layoutvlm: Differentiable optimization of 3d layout via vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 29469-29478.

Wang, Y.; Xian, Z.; Chen, F.; Wang, T.-H.; Wang, Y.; Fragkiadaki, K.; Erickson, Z.; Held, D.; and Gan, C. 2023. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. arXiv preprint arXiv:2311.01455.

Yang, Y.; Sun, F.-Y.; Weihs, L.; VanderBilt, E.; Herrasti, A.; Han, W.; Wu, J.; Haber, N.; Krishna, R.; Liu, L.; et al. 2024. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16227-16237.

Zhou, X.; et al. 2024. Genesis: A Generative and Universal Physics Engine for Robotics and Beyond. https: //github.com/Genesis-Embodied-AI/Genesis.

## Appendix

Prompts used with the environment generator \( G \) and analysis generator \( F \) as described in the experimental setup.

Prompt used for \( F \) to analyze the top-down view of the agent’s trajectory

Analyze the robot navigation paths shown in this floor plan and provide:

- Success assessment: Did the robot complete its navigation task successfully?

- Intermediate concerns: What issues or risks occurred during navigation (e.g., clos

proximity to obstacles, narrow clearances, suboptimal routing)?

- Targeted adaptations: What specific improvements or training scenarios should be

implemented to address these concerns?

The following prompt is used with the current environment's top-down view to guide the environment generator \( G \) in moving individual objects.

Prompt used for \( G \) to perturb individual objects

---

Analysis of agent's trajectory.

			[analysis from F]

Given the analysis of the agent's trajectory, make the environment more challenging

	for object navigation.

Select ONE object visible in the scene and propose a single move (change in \( x, y \)

	coordinates) that:

	- Creates navigation challenges based on the analysis suggestions

	- Maintains realism (objects in plausible locations)

- Stays within apartment bounds

Use the `propose_move_instruction` function to specify the move.

	Constraints:

	- Apartment uses normalized 100x100 grid (x:[0,100], y:[0,100])

	- Example: sofa at x=27, y=5, rotation=0

	- Keep objects inside walls

- Don't block doorways completely (but can reduce clearances)

	- Avoid overlapping with other objects

---

The following function schema is used with the above prompt and passed as a tool to the OpenAI API. Note that movements tre specified using directional commands (left/right for x-axis, up/down for y-axis) rather than absolute coordinates.

Function schema used with the above prompt with \( G \)

---

perturb_instruction_tools = [ \{

	"type": "function",

	"name": "propose_move_instruction",

	"description": (   )

		"Propose the change in position for exactly one object. "

		"Movement must be expressed in a normalized 100x100 apartment grid as

left/right and up/down units."),

	"parameters": \{

		"type": "object",

		"properties": \{

			"object_id": \{"type": "string", "enum": allowed_object_ids\},

			"x_direction": \{"type": "string", "enum": ["left", "right"]\},

			"x_units": \{"type": "number", "minimum": 0, "maximum": 100\},

			"y_direction": \{"type": "string", "enum": ["up", "down"]\},

			"y_units": \{"type": "number", "minimum": 0, "maximum": 100\},

			"rotation": \{"type": "number", "minimum": 0, "maximum": 360\}\},

		"required": ["object_id", "x_direction", "x_units", "y_direction", "y_units",

"rotation"],

		"additionalProperties": False

	\}\}]

---