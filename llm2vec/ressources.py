# Test data with clear topical separation
instruction = "Given a web search query, retrieve relevant passages that answer the query:"

queries = [
    [instruction, "What makes a good poem?"],  # Poetry-focused query
    [instruction, "What is quantum entanglement?"]  # Science-focused query
]

# Let's create documents with clear topical separation
poetry_documents = [
    "Poetry is the rhythmical creation of beauty in words, using metaphor and imagery to evoke emotions.",
    "A sonnet is a fourteen-line poem written in iambic pentameter, often exploring themes of love.",
    "Free verse poetry abandons traditional rhyme and meter for a more natural flow of language.",
    "Haiku is a Japanese poetry form with three lines of 5, 7, and 5 syllables, often about nature.",
    "Metaphors in poetry create powerful connections by comparing unlike things in creative ways.",
    "The use of alliteration in poetry creates musical effects through repeated consonant sounds.",
    "Romantic poets like Wordsworth emphasized emotional expression and connection with nature.",
    "Modern poetry often breaks traditional rules to experiment with form and meaning.",
    "Imagery in poetry uses sensory details to create vivid mental pictures for readers.",
    "Poetry can transform ordinary experiences into extraordinary insights through careful word choice.",
    "Rhythm in poetry creates a musical quality that enhances the emotional impact of words.",
    "Symbolism in poetry uses objects or ideas to represent deeper meanings and themes.",
    "Epic poetry tells grand stories of heroic deeds through extended narrative verse.",
    "The power of poetry lies in its ability to compress complex emotions into precise language.",
    "Line breaks in poetry control pacing and create emphasis for important words.",
    "Poetry workshops focus on crafting powerful imagery and emotional resonance.",
    "Contemporary poets often explore themes of identity and social justice in their work.",
    "The best poems leave readers with new perspectives on familiar experiences.",
    "Poetry slams showcase the performative aspects of spoken word poetry.",
    "Poetic devices like assonance create musical patterns through vowel sounds.",
    "Writing poetry requires attention to both sound and meaning in language.",
    "Poetry anthologies collect diverse voices and styles in one volume.",
    "Concrete poetry uses visual arrangement of words to enhance meaning.",
    "Poetry therapy uses creative expression for emotional healing.",
    "The study of poetry enhances appreciation of language and meaning."
]

science_documents = [
    "Quantum entanglement occurs when particles become correlated in ways that can't be explained by classical physics.",
    "The uncertainty principle states that we cannot simultaneously know both position and momentum precisely.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape, not even light.",
    "DNA carries genetic information in a double helix structure made of nucleotide base pairs.",
    "Einstein's theory of relativity shows that space and time are interconnected dimensions.",
    "Photosynthesis converts light energy into chemical energy in plants using chlorophyll.",
    "Neutrinos are nearly massless particles that rarely interact with ordinary matter.",
    "The strong nuclear force holds atomic nuclei together despite electromagnetic repulsion.",
    "Dark matter makes up about 27% of the universe but cannot be directly observed.",
    "Quantum superposition allows particles to exist in multiple states simultaneously.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "String theory proposes that all particles are actually tiny vibrating strings.",
    "Chemical bonds form when atoms share or transfer electrons.",
    "The second law of thermodynamics states that entropy always increases in closed systems.",
    "Quantum tunneling allows particles to pass through energy barriers.",
    "The Higgs boson gives other particles their mass through the Higgs field.",
    "Neural networks in the brain process information through synaptic connections.",
    "Dark energy causes the accelerating expansion of the universe.",
    "Quantum computers use qubits that can exist in multiple states at once.",
    "The periodic table organizes elements by their atomic properties.",
    "RNA acts as a messenger molecule in protein synthesis.",
    "Fusion reactions power stars by combining light atomic nuclei.",
    "The electromagnetic spectrum includes all wavelengths of light.",
    "Cellular respiration converts glucose into energy in mitochondria.",
    "Wave-particle duality shows that light behaves as both waves and particles."
]

