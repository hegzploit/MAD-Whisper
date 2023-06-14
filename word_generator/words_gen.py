# Load the replacement table from a file
replacement_table = {}
with open("replacement_table.txt", "r") as f:
    for line in f:
        letter, replacements = line.strip().split(":")
        replacement_table[letter] = replacements.split()

# Load a list of words from a file words.txt into a comma-separated string
words = []
with open("words.txt", "r") as f:
    for line in f:
        words.append(line.strip())

wrong_words = []
for word in words:
    for letter, replacements in replacement_table.items():
        if letter in word:
            for replacement in replacements:
                wrong_words.append(word.replace(letter, replacement) + ":" + word)

# Save the list of wrong words to a file
with open("generated_words.txt", "w") as f:
    for word in wrong_words:
        f.write(word + "\n")
