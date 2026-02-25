Architecture overview
=====================

This document describes the high-level architecture of the DARTS devkit.
The system is designed around a **plugin-based architecture** that allows
core dataset functionality to remain lightweight, while extensions can be developed and distributed independently.

Architecture Schema
-------------------
.. mermaid::

   classDiagram
       direction TB

       %% =====================
       %% Core package
       %% =====================
       class DARTS {
           +Path root
           +str version
       }

       %% =====================
       %% Interfaces
       %% =====================

       class EvaluatorInterface {
           <<abstract>>
           +evaluate(darts: DARTS)
       }

       %% =====================
       %% Registries
       %% =====================

       class EvaluatorRegistry {
           +register(name, cls: EvaluateInterface)
           +get(name)
           +available()
       }

       %% =====================
       %% Evaluation plugins
       %% =====================
       class EvalOne {
           +evaluate(darts: DARTS)
       }

       class EvalTwo {
           +evaluate(darts: DARTS)
       }

       %% =====================
       %% Relationships
       %% =====================

       EvaluatorInterface <|-- EvalOne
       EvaluatorInterface <|-- EvalTwo

       EvaluatorRegistry --> EvalOne : registers
       EvaluatorRegistry --> EvalTwo : registers

       EvalOne --> DARTS : uses
       EvalTwo --> DARTS : uses

Architecture Principles
-----------------------

The architecture follows these key principles:

- **Separation of concerns**  
  The core ``DARTS`` class is responsible only for dataset access, indexing,
  and querying.

- **Plugin-based extensibility**  
  Additional logic is implemented in separate packages
  that register themselves via registries.

- **Optional dependencies**  
  Users may install only the plugins they need (e.g. a specific visualizer
  or evaluation protocol).

- **Runtime selection**  
  Implementations are selected by name at runtime using registries, without
  conditional imports in the core code.

Example Usage
-------------

A typical usage pattern looks like this:

.. code-block:: python

    import logging
    import darts.evaluation as de

    from darts import DARTS, EvaluateRegistry
    logger = logging.getLogger(__name__)


    def main() -> None:
        """Main method for testing purposes."""
        #logging.basicConfig(level=logging.ERROR)
        logging.basicConfig(level=logging.INFO)
        darts = DARTS("/data/dataset", "dataset_exported")
        darts.verify_integrity()
        logger.info(darts.category.all())
        logger.info(darts)
        logger.info(EvaluateRegistry.available())
        de.register_eval_one()  # registers only EvalOne
        logger.info(EvaluateRegistry.available())
        evaluator_cls = EvaluateRegistry.get("EvalOne")
        evaluator = evaluator_cls()
        logger.info(evaluator.evaluate(darts))


    if __name__ == "__main__":
        main()

This approach allows new visualization or evaluation backends to be added
without changes to the DARTS core.