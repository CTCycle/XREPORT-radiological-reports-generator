# README Writing Guidelines

This document defines the **required structure, scope, and writing standards** for producing a proper README file.  
Its purpose is to ensure clarity, consistency, and usability across projects, especially for applications involving backend and frontend components or machine learning workflows.

A proper README explains **what the project does, how to install it, and how to use it**, without exposing internal code details or overwhelming the reader.

If a section does not fit in the observed project, skip it completely and adapt section numbering accordingly.

---

## 1. Project Overview

This section provides a concise but comprehensive description of the project.

It must clearly explain:
- The purpose of the software.
- The problem it aims to solve.
- The general mechanism or method used.
- The high-level architecture of the system.

If the project includes both backend and frontend components, briefly describe their roles and how they interact.  
Do not describe internal code elements such as functions, classes, or modules.

The reader should understand **what the system does and how it is organized**, without reading the source code.

---

## 2. Model and Dataset (Optional)

This section is required only for machine learning projects.

Describe at a high level:
- The type of model or algorithm used.
- The learning paradigm, such as supervised learning or reinforcement learning.
- The nature and origin of the dataset used for training or evaluation.

Do not include architectural details, hyperparameters, or implementation specifics.  
If datasets are user-provided, synthetic, or externally sourced, state this clearly.

If model or dataset details are uncertain, this must be explicitly acknowledged.

---

## 3. Installation

This section explains how to install and prepare the application for use.

Instructions must be:
- Minimal
- Reproducible
- Focused on outcomes rather than internal mechanics

Avoid documenting every internal setup step unless strictly necessary.

### 3.1 Windows (One Click Setup)

If the project provides an automated Windows setup:
- State clearly that the setup is automated.
- List, in order, what the launcher performs at a high level.
- Explain first-run behavior versus subsequent runs.
- Specify whether the installation is portable or modifies the host system.

Do not describe script internals.

### 3.2 macOS / Linux (Manual Setup)

If manual setup is required:
- List prerequisites explicitly.
- Provide numbered installation steps.
- Separate backend and frontend setup if applicable.
- Mention optional components only when relevant.

Terminal commands may be shown in fenced blocks, but should remain concise.

---

## 4. How to Use

This section explains how users interact with the application after installation.

### 4.1 Windows

Describe:
- How to launch the application.
- The URL or interface where the application becomes available.

### 4.2 macOS / Linux

Provide:
- Separate commands for backend and frontend if applicable.
- The local URLs for the UI, backend API, and documentation if exposed.

### 4.3 Using the Application

Describe the **operational workflow**, not the internal logic.

Examples of acceptable topics:
- Loading or preparing data.
- Running training, analysis, or processing tasks.
- Executing inference or simulations.
- Reviewing outputs, logs, or stored results.

If the application includes a UI, include linked screenshots from the project’s assets directory, each accompanied by a short functional description.

---

## 5. Setup and Maintenance

Describe any maintenance or utility scripts provided with the project.

List available actions in bullet form, each with a short explanation, such as:
- Clearing logs.
- Resetting application state.
- Reinitializing databases.
- Removing local installations.

Focus on **what each action does**, not how it is implemented.

---

## 6. Resources

Explain the purpose of the project’s resource or data directory.

For each subdirectory:
- Begin with the directory name followed by a colon.
- Describe what it contains.
- Explain how it is used by the application.

If templates or sample files exist, note their location and intended use.

---

## 7. Configuration

Describe where configuration files are located and how they are applied.

If both backend and frontend configurations exist:
- Describe them separately.
- Clarify how configuration is loaded, for example via environment variables or configuration files.

Include a configuration table with the following format:

| Variable | Description |
|----------|-------------|
| VARIABLE_NAME | Purpose, definition location, and default value |

Each row must specify:
- The variable name.
- Its function.
- Where it is defined.
- Its default behavior or value.

---

## 8. License

State the license type clearly and refer to the LICENSE file for full terms.

---

## Final Notes

A proper README:
- Is written for users, not developers.
- Explains functionality and workflow, not code.
- Is factual and avoids speculative claims.
- Is structured, readable, and skimmable.

Any README written using these guidelines must follow this structure exactly.
