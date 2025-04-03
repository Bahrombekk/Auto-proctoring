# src/db_management.py
import lmdb
import os
import shutil
import pickle

class DBManager:
    def __init__(self, lmdb_dir="lmdb_data"):
        self.lmdb_dir = lmdb_dir
    
    def list_users(self):
        try:
            if not os.path.exists(self.lmdb_dir):
                print(f"{self.lmdb_dir} mavjud emas!")
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
                print("Baza bo'sh!")
            else:
                print("Bazadagi foydalanuvchilar:")
                for user in users:
                    print(f"- {user}")
            return users
        except Exception as e:
            print(f"Xato yuz berdi: {e}")
            return []

    def backup_lmdb(self, backup_dir="lmdb_backup"):
        try:
            if not os.path.exists(self.lmdb_dir):
                print(f"{self.lmdb_dir} mavjud emas!")
                return False
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(self.lmdb_dir, backup_dir)
            print(f"Baza {backup_dir} ga zaxiralandi.")
            return True
        except Exception as e:
            print(f"Zaxiralashda xato: {e}")
            return False

    def delete_user(self, user_id):
        try:
            if not os.path.exists(self.lmdb_dir):
                print(f"{self.lmdb_dir} mavjud emas!")
                return False
            env = lmdb.open(self.lmdb_dir, map_size=10485760)
            with env.begin(write=True) as txn:
                key = str(user_id).encode('utf-8')
                if txn.get(key) is None:
                    print(f"ID: {user_id} bazada topilmadi!")
                    env.close()
                    return False
                txn.delete(key)
                print(f"ID: {user_id} bazadan o'chirildi.")
            env.close()
            return True
        except Exception as e:
            print(f"O'chirishda xato: {e}")
            return False

    def delete_all_users(self):
        try:
            if not os.path.exists(self.lmdb_dir):
                print(f"{self.lmdb_dir} mavjud emas!")
                return False
            env = lmdb.open(self.lmdb_dir, map_size=10485760)
            with env.begin(write=True) as txn:
                cursor = txn.cursor()
                count = 0
                for key, _ in cursor:
                    txn.delete(key)
                    count += 1
                if count == 0:
                    print("Baza allaqachon bo'sh!")
                else:
                    print(f"Barcha {count} ta foydalanuvchi o'chirildi.")
            env.close()
            return True
        except Exception as e:
            print(f"Barchasini o'chirishda xato: {e}")
            return False

    def search_user(self, user_id):
        try:
            if not os.path.exists(self.lmdb_dir):
                print(f"{self.lmdb_dir} mavjud emas!")
                return None
            env = lmdb.open(self.lmdb_dir, readonly=True)
            with env.begin() as txn:
                key = str(user_id).encode('utf-8')
                value = txn.get(key)
                if value is None:
                    print(f"ID: {user_id} topilmadi!")
                    env.close()
                    return None
                embedding = pickle.loads(value)
                print(f"ID: {user_id} topildi. Embedding uzunligi: {len(embedding)}")
                print(f"Embedding namunasi: {embedding[:5]}")
                env.close()
                return embedding
        except Exception as e:
            print(f"Qidirishda xato: {e}")
            return None

    def run_menu(self):
        while True:
            choice = self.show_menu()
            if choice == "1":
                self.list_users()
            elif choice == "2":
                self.backup_lmdb("lmdb_backup")
            elif choice == "3":
                user_id = input("O'chiriladigan foydalanuvchi ID sini kiriting: ")
                self.delete_user(user_id)
            elif choice == "4":
                confirm = input("Barcha foydalanuvchilarni o'chirishni tasdiqlaysizmi? (ha/yo'q): ")
                if confirm.lower() == "ha":
                    self.delete_all_users()
                else:
                    print("O'chirish bekor qilindi.")
            elif choice == "5":
                user_id = input("Qidiriladigan foydalanuvchi ID sini kiriting: ")
                self.search_user(user_id)
            elif choice == "0":
                print("Dastur yakunlandi.")
                break
            else:
                print("Noto'g'ri tanlov! Iltimos, 0-5 oralig'ida raqam kiriting.")
    
    def show_menu(self):
        print("\n=== LMDB Baza Boshqaruvi ===")
        print("1. Bazadagi foydalanuvchilar ro'yxatini ko'rish")
        print("2. Bazani zaxiralash")
        print("3. Foydalanuvchini o'chirish")
        print("4. Barcha foydalanuvchilarni o'chirish")
        print("5. Foydalanuvchini qidirish")
        print("0. Chiqish")
        choice = input("Tanlovni kiriting (0-5): ")
        return choice

if __name__ == "__main__":
    db_manager = DBManager("lmdb_data")
    db_manager.run_menu()