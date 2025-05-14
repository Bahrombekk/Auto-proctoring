import lmdb
import os
import shutil
import pickle
from config import LMDB_DIR
from src.face_database import FaceDatabase
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DBManager:
    def __init__(self, lmdb_dir=LMDB_DIR):
        self.lmdb_dir = lmdb_dir
        self.face_db = FaceDatabase(lmdb_dir)  # FaceDatabase ni ishlatish uchun
    
    def add_user(self, user_id, image_path):
        """Foydalanuvchi ID va rasm yo'lini qabul qilib, yuz embeddingini LMDB ga saqlaydi."""
        try:
            if not os.path.exists(image_path):
                logging.error(f"Rasm fayli topilmadi: {image_path}")
                return False
            
            # FaceDatabase yordamida yuz embeddingini saqlash
            success = self.face_db.save_to_lmdb(image_path, user_id)
            if success:
                logging.info(f"Foydalanuvchi ID: {user_id} muvaffaqiyatli qo'shildi.")
                return True
            else:
                logging.error(f"Foydalanuvchi qo'shishda xato: ID {user_id}")
                return False
        except Exception as e:
            logging.error(f"Foydalanuvchi qo'shishda xato: {e}")
            return False

    def list_users(self):
        try:
            if not os.path.exists(self.lmdb_dir):
                logging.error(f"{self.lmdb_dir} does not exist!")
                return []
            env = lmdb.open(self.lmdb_dir, readonly=True)
            users = []
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    user_id = key.decode('utf-8')
                    users.append(user_id)
            env.close()
            if not users:
                logging.info("Database is empty!")
            else:
                logging.info("Users in database:")
                for user in users:
                    logging.info(f"- {user}")
            return users
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return []

    def backup_lmdb(self, backup_dir="lmdb_backup"):
        try:
            if not os.path.exists(self.lmdb_dir):
                logging.error(f"{self.lmdb_dir} does not exist!")
                return False
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(self.lmdb_dir, backup_dir)
            logging.info(f"Database backed up to {backup_dir}.")
            return True
        except Exception as e:
            logging.error(f"Backup error: {e}")
            return False

    def delete_user(self, user_id):
        try:
            if not os.path.exists(self.lmdb_dir):
                logging.error(f"{self.lmdb_dir} does not exist!")
                return False
            env = lmdb.open(self.lmdb_dir, map_size=10485760)
            with env.begin(write=True) as txn:
                key = str(user_id).encode('utf-8')
                if txn.get(key) is None:
                    logging.error(f"ID: {user_id} not found in database!")
                    env.close()
                    return False
                txn.delete(key)
                logging.info(f"ID: {user_id} deleted from database.")
            env.close()
            return True
        except Exception as e:
            logging.error(f"Deletion error: {e}")
            return False

    def delete_all_users(self):
        try:
            if not os.path.exists(self.lmdb_dir):
                logging.error(f"{self.lmdb_dir} does not exist!")
                return False
            env = lmdb.open(self.lmdb_dir, map_size=10485760)
            with env.begin(write=True) as txn:
                cursor = txn.cursor()
                count = 0
                for key, _ in cursor:
                    txn.delete(key)
                    count += 1
                if count == 0:
                    logging.info("Database is already empty!")
                else:
                    logging.info(f"All {count} users deleted.")
            env.close()
            return True
        except Exception as e:
            logging.error(f"Error deleting all users: {e}")
            return False

    def search_user(self, user_id):
        try:
            if not os.path.exists(self.lmdb_dir):
                logging.error(f"{self.lmdb_dir} does not exist!")
                return None
            env = lmdb.open(self.lmdb_dir, readonly=True)
            with env.begin() as txn:
                key = str(user_id).encode('utf-8')
                value = txn.get(key)
                if value is None:
                    logging.error(f"ID: {user_id} not found!")
                    env.close()
                    return None
                embedding = pickle.loads(value)
                logging.info(f"ID: {user_id} found. Embedding length: {len(embedding)}")
                logging.info(f"Embedding sample: {embedding[:5]}")
                env.close()
                return embedding
        except Exception as e:
            logging.error(f"Search error: {e}")
            return None

    def run_menu(self):
        while True:
            choice = self.show_menu()
            if choice == "1":
                self.list_users()
            elif choice == "2":
                self.backup_lmdb("lmdb_backup")
            elif choice == "3":
                user_id = input("Enter user ID to delete: ")
                self.delete_user(user_id)
            elif choice == "4":
                confirm = input("Confirm deletion of all users? (yes/no): ")
                if confirm.lower() == "yes":
                    self.delete_all_users()
                else:
                    logging.info("Deletion cancelled.")
            elif choice == "5":
                user_id = input("Enter user ID to search: ")
                self.search_user(user_id)
            elif choice == "6":
                user_id = input("Enter user ID to add: ")
                image_path = input("Enter path to user image: ")
                self.add_user(user_id, image_path)
            elif choice == "0":
                logging.info("Program terminated.")
                break
            else:
                logging.warning("Invalid choice! Please enter 0-6.")

    def show_menu(self):
        print("\n=== LMDB Database Management ===")
        print("1. List all users")
        print("2. Backup database")
        print("3. Delete a user")
        print("4. Delete all users")
        print("5. Search for a user")
        print("6. Add a user")
        print("0. Exit")
        choice = input("Enter choice (0-6): ")
        return choice

if __name__ == "__main__":
    db_manager = DBManager()
    db_manager.run_menu()