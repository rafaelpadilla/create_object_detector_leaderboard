try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    print("Error importing dotenv")
    
__version__ = "0.0.3"
