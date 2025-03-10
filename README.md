# AI Search Algorithms: Harry Potter-themed Problem Solving

## Overview
This repository contains my solutions to three assignments from a university AI course, each exploring different problem-solving approaches in artificial intelligence through a Harry Potter-themed framework. The assignments progress from deterministic search to logical inference and finally to stochastic environments.

## Problem Domains

### Part 1: Deterministic Search
In this assignment, we control a group of wizards to find and destroy Voldemort's horcruxes without casualties. The problem involves navigating a grid world with:
- Multiple wizard agents with limited lives
- Death eaters moving in predefined patterns
- Horcruxes to be destroyed
- Voldemort who can only be defeated after destroying all horcruxes

**Techniques implemented:**
- A* search algorithm
- Problem modeling (state representation, actions, transitions)
- Admissible heuristic design

### Part 2: Logical Inference
This assignment follows Harry Potter as he attempts to break into Gringotts bank to find a deathly hallow. The challenge involves:
- Logical inference to detect traps
- Knowledge representation about the environment
- Incremental discovery of the map
- Strategic navigation around dragons and traps

**Techniques implemented:**
- Knowledge base construction and updating
- Logical inference rules
- Planning with incomplete information

### Part 3: Stochastic Task
Building on part 1, this assignment extends the wizard-horcrux problem to a stochastic environment where:
- Death eaters move probabilistically along their paths
- Horcruxes can randomly change locations
- Points are awarded for destroying horcruxes and deducted for hazards
- Decisions must optimize expected utility

**Techniques implemented:**
- Decision-making under uncertainty
- Value iteration
- Markov Decision Processes

## Repository Structure
```
├── hw1-deterministic-search/
│   ├── README.md - Problem details and solution approach
│   ├── ex1.py - Core implementation
│   ├── check.py - Testing framework
│   └── util files
├── hw2-logical-inference/
│   ├── README.md - Problem details and solution approach
│   ├── ex2.py - Core implementation
│   ├── check.py - Testing framework
│   └── util files
└── hw3-stochastic-task/
    ├── README.md - Problem details and solution approach
    ├── ex3.py - Core implementation
    ├── check.py - Testing framework
    └── util files
```

## Key Learnings
Through these assignments, I've gained experience in:
- Modeling complex problems as search problems
- Implementing and optimizing search algorithms
- Designing efficient heuristics
- Reasoning with logical inference
- Making optimal decisions under uncertainty
- Balancing completeness, optimality, and efficiency

## Technologies Used
- Python 3.10

