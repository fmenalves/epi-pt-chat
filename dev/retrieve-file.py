import glob
import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

file_path = "log.txt"
file = open(file_path, "a")
# file.close()


chrome_options = Options()
# Create a CSV file
csv_file = open("./finaldbpt-dev.csv", "a")

# Write header to the CSV file
csv_file.write("MedID,Name,Subs,FF,Dosage,TAIM,NewFilename,timestamp\n")

# Get the current timestamp in string format

# Open the CSV file
download_folder = "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final"
chrome_options.add_experimental_option(
    "prefs",
    {
        "plugins.always_open_pdf_externally": True,  # Automatically download PDFs
        "download.default_directory": download_folder,  # Set your download directory
        "profile.default_content_settings.popups": 0,
        "download.prompt_for_download": False,
    },
)


driver = webdriver.Chrome(options=chrome_options)


def name_in_filenames(directory, search_name, product_name):
    """
    Check if 'search_name' exists in any part of the filenames within 'directory'.

    :param directory: The directory to search in.
    :param search_name: The name (substring) to search for within filenames.
    :return: True if 'search_name' is found in any filename, False otherwise.
    """
    # List all files in the directory
    files_in_directory = os.listdir(directory)
    # print(product_name)
    # Check if 'search_name' is a substring of any filename
    for filename in files_in_directory:
        if search_name in filename:
            return True, filename
    #        if product_name.lower() + "-epar-product-information" in filename:
    #            print(product_name.lower() + "-epar-product-information" in filename)
    #            return True, filename
    #        elif (
    #            product_name.lower().split(" ")[0] + "-epar-product-information" in filename
    #        ):
    #            return True, filename
    return False, search_name


def get_last_filename_and_rename(save_folder, new_filename, override=False):
    files = glob.glob(save_folder + "/*.pdf")
    max_file = max(files, key=os.path.getctime)
    filename = max_file.split("/")[-1].replace(".pdf", "")
    # print(filename)
    # print(max_file, filename, new_filename)
    new_path = max_file.replace(filename, new_filename)
    if "documento" in max_file and "pdf" in max_file:
        #    print(max_file, new_path)
        os.rename(max_file, new_path)
    elif "epar-product-information_pt" in max_file and "pdf" in max_file:
        # pass
        os.rename(max_file, new_path)
    elif override:
        os.rename(max_file, new_path)

    else:
        file.write("error on rename: max file: " + max_file + " new: " + new_filename)
        print("error on rename?", max_file, new_filename)
    # print(new_path)
    return new_path


def retrieve_all(
    driver,
):
    # browser = webdriver.Chrome("")  # searchs path
    driver.get("https://extranet.infarmed.pt/INFOMED-fo/pesquisa-avancada.xhtml")
    # elem = WebDriverWait(driver, 1).until(
    elem = WebDriverWait(driver, 1).until(
        EC.presence_of_element_located((By.ID, "mainForm:estado-comercializacao_label"))
    )

    driver.execute_script("arguments[0].scrollIntoView(true);", elem)

    elem.click()

    actions = ActionChains(driver)

    # Move to the element, then move by an offset, and click
    # For example, to move 10 pixels down from the element
    offset_x = 0  # No horizontal movement
    offset_y = 100  # Move 10 pixels down
    actions.move_to_element(elem).move_by_offset(offset_x, offset_y).click().perform()

    elem = WebDriverWait(driver, 1).until(
        EC.presence_of_element_located((By.ID, "mainForm:btnDoSearch"))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    elem.click()

    time.sleep(6)

    # Click on the element

    total = driver.find_element(By.CLASS_NAME, "ui-paginator-current")

    print(total.get_attribute("textContent"))

    # get number per page
    select = driver.find_element(By.NAME, "mainForm:dt-medicamentos_rppDD")
    #
    select_obj = Select(select)
    select_obj.select_by_value("100")

    for i in range(81, 93):
        print("page nr: " + str(i) + " " + "--" * 40)
        pages_all = driver.find_element(By.CLASS_NAME, "ui-paginator-pages")
        try:
            page = pages_all.find_element(By.XPATH, f".//*[contains(text(), '{i}')]")
        except:
            print("failed to change page")
            new_i = 10
            while new_i < i:
                pages_all = driver.find_element(By.CLASS_NAME, "ui-paginator-pages")

                print("seraching for page: " + str(new_i))
                page = pages_all.find_element(
                    By.XPATH, f".//*[contains(text(), '{new_i}')]"
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", page)

                page.click()
                new_i += 4
                print("new_i: " + str(new_i))
                time.sleep(5)
            page = pages_all.find_element(By.XPATH, f".//*[contains(text(), '{i}')]")

        driver.execute_script("arguments[0].scrollIntoView(true);", page)
        page.click()
        time.sleep(5)
        print("starting...")
        tds = driver.find_elements(By.CLASS_NAME, "coluna-documento")
        for idx, td in enumerate(tds):
            if td.get_attribute("role") == "gridcell":
                name = td.find_element(By.XPATH, "./ancestor::tr/td[2]")
                dosage = td.find_element(By.XPATH, "./ancestor::tr/td[5]")
                ff = td.find_element(By.XPATH, "./ancestor::tr/td[4]")
                subs = td.find_element(By.XPATH, "./ancestor::tr/td[3]")
                medid = td.find_element(By.XPATH, "./ancestor::tr/td[1]")
                taim = td.find_element(By.XPATH, "./ancestor::tr/td[6]")
                timestamp = time.strftime("%Y%m%d%H%M%S")
                # print(medid.get_attribute("role"))
                # print(medid.get_attribute("value"))
                # print(medid.get_attribute("text"))
                # print(medid.get_attribute("textContent"))

                #  print(medid.text)
                name_content = name.text
                dosagetext = dosage.text
                fftext = ff.text
                newfilename = name_content + str(fftext) + str(dosagetext)
                newfilename = (
                    newfilename.replace("/", "_")
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "_")
                    .replace(".", "_")
                    .replace("-", "_")
                )

                existing, filenametostore = name_in_filenames(
                    download_folder, newfilename, name_content
                )
                csvline = (
                    medid.get_attribute("textContent")
                    + ',"'
                    + name_content
                    + '","'
                    + subs.text
                    + '","'
                    + ff.text
                    + '","'
                    + dosagetext
                    + '","'
                    + taim.text
                    + '","'
                    + filenametostore
                    + '",'
                    + timestamp
                )
                # Write data to the CSV file
                csv_file.write(csvline + "\n")

                if not existing:
                    print(newfilename + " does not exist: - Downloading")
                    driver.execute_script("arguments[0].scrollIntoView(true);", td)

                    ###get the product name for naming the pdf.
                    tags = td.find_elements(By.TAG_NAME, "a")
                    try:
                        for a in tags:
                            if "pesqAvancadaDatableRcmIcon" in a.get_attribute(
                                "id"
                            ) or "esqAvancadaDatableEmaIcon" in a.get_attribute("id"):
                                #  print(a.get_attribute("id"))

                                driver.execute_script(
                                    "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center', inline: 'nearest'});",
                                    a,
                                )
                                time.sleep(3)

                                # Step 1: Get current window handles
                                original_window = driver.current_window_handle
                                original_window_handles = driver.window_handles
                                clickable_element = WebDriverWait(driver, 20).until(
                                    EC.element_to_be_clickable(a)
                                )
                                a.click()
                                time.sleep(3)
                                get_last_filename_and_rename(
                                    download_folder, newfilename
                                )
                                # Step 3: Check if a new tab has opened
                                new_window_handles = [
                                    handle
                                    for handle in driver.window_handles
                                    if handle not in original_window_handles
                                ]

                                if new_window_handles:
                                    # Assuming the first new window handle is the tab we want to close
                                    new_tab_handle = new_window_handles[0]

                                    # Step 4: Switch to the new tab and close it
                                    driver.switch_to.window(new_tab_handle)
                                    driver.close()

                                    # Step 5: Switch back to the original window/tab
                                    driver.switch_to.window(original_window)

                    except Exception as err:
                        print("Error " + name_content, err)
                        file.write("Error " + name_content + "\n")
                        # driver.switch_to.window(driver.window_handles[-1])
                else:
                    print(newfilename + " already exists")

    # Close the CSV file
    csv_file.close()
    file.close()


def retrieve_by_query(driver, query, query_type="medication"):
    driver.get("https://extranet.infarmed.pt/INFOMED-fo/pesquisa-avancada.xhtml")

    # elem = WebDriverWait(driver, 1).until(
    input_field = WebDriverWait(driver, 1).until(
        EC.presence_of_element_located((By.ID, "mainForm:medicamento_input"))
    )

    driver.execute_script("arguments[0].scrollIntoView(true);", input_field)
    # Input text into the field
    time.sleep(0.5)

    input_field.send_keys(query)
    # print("ja enviou??")
    time.sleep(6)

    # Optional: Submit the form if needed
    input_field.send_keys(Keys.RETURN)  # Or use other methods to submit if applicable

    time.sleep(6)

    total = driver.find_element(By.CLASS_NAME, "ui-paginator-current")

    print(total.get_attribute("textContent"))

    # get number per page
    select = driver.find_element(By.NAME, "mainForm:dt-medicamentos_rppDD")
    #
    select_obj = Select(select)
    select_obj.select_by_value("100")

    for i in range(1, 19):
        print("page nr: " + str(i) + " " + "--" * 40)
        pages_all = driver.find_element(By.CLASS_NAME, "ui-paginator-pages")
        try:
            page = pages_all.find_element(By.XPATH, f".//*[contains(text(), '{i}')]")
        except:
            print("failed to change page")
            new_i = 10
            while new_i < i:
                pages_all = driver.find_element(By.CLASS_NAME, "ui-paginator-pages")

                print("seraching for page: " + str(new_i))
                page = pages_all.find_element(
                    By.XPATH, f".//*[contains(text(), '{new_i}')]"
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", page)

                page.click()
                new_i += 4
                print("new_i: " + str(new_i))
                time.sleep(5)
            page = pages_all.find_element(By.XPATH, f".//*[contains(text(), '{i}')]")

        driver.execute_script("arguments[0].scrollIntoView(true);", page)
        page.click()
        time.sleep(5)
        print("starting...")
        tds = driver.find_elements(By.CLASS_NAME, "coluna-documento")
        for idx, td in enumerate(tds):
            if td.get_attribute("role") == "gridcell":
                name = td.find_element(By.XPATH, "./ancestor::tr/td[2]")
                dosage = td.find_element(By.XPATH, "./ancestor::tr/td[5]")
                ff = td.find_element(By.XPATH, "./ancestor::tr/td[4]")
                subs = td.find_element(By.XPATH, "./ancestor::tr/td[3]")
                medid = td.find_element(By.XPATH, "./ancestor::tr/td[1]")
                taim = td.find_element(By.XPATH, "./ancestor::tr/td[6]")
                timestamp = time.strftime("%Y%m%d%H%M%S")
                # print(medid.get_attribute("role"))
                # print(medid.get_attribute("value"))
                # print(medid.get_attribute("text"))
                # print(medid.get_attribute("textContent"))

                #  print(medid.text)
                name_content = name.text
                dosagetext = dosage.text
                fftext = ff.text
                newfilename = name_content + str(fftext) + str(dosagetext)
                newfilename = (
                    newfilename.replace("/", "_")
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "_")
                    .replace(".", "_")
                    .replace("-", "_")
                )

                existing, filenametostore = name_in_filenames(
                    download_folder, newfilename, name_content
                )
                csvline = (
                    medid.get_attribute("textContent")
                    + ',"'
                    + name_content
                    + '","'
                    + subs.text
                    + '","'
                    + ff.text
                    + '","'
                    + dosagetext
                    + '","'
                    + taim.text
                    + '","'
                    + filenametostore
                    + '",'
                    + timestamp
                )
                # Write data to the CSV file
                csv_file.write(csvline + "\n")

                if not existing:
                    print(newfilename + " does not exist: - Downloading")
                    driver.execute_script("arguments[0].scrollIntoView(true);", td)

                    ###get the product name for naming the pdf.
                    tags = td.find_elements(By.TAG_NAME, "a")
                    try:
                        for a in tags:
                            if "pesqAvancadaDatableRcmIcon" in a.get_attribute(
                                "id"
                            ) or "esqAvancadaDatableEmaIcon" in a.get_attribute("id"):
                                #  print(a.get_attribute("id"))

                                driver.execute_script(
                                    "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center', inline: 'nearest'});",
                                    a,
                                )
                                time.sleep(3)

                                # Step 1: Get current window handles
                                original_window = driver.current_window_handle
                                original_window_handles = driver.window_handles
                                clickable_element = WebDriverWait(driver, 20).until(
                                    EC.element_to_be_clickable(a)
                                )
                                a.click()
                                time.sleep(3)
                                get_last_filename_and_rename(
                                    download_folder, newfilename, override=True
                                )
                                # Step 3: Check if a new tab has opened
                                new_window_handles = [
                                    handle
                                    for handle in driver.window_handles
                                    if handle not in original_window_handles
                                ]

                                if new_window_handles:
                                    # Assuming the first new window handle is the tab we want to close
                                    new_tab_handle = new_window_handles[0]

                                    # Step 4: Switch to the new tab and close it
                                    driver.switch_to.window(new_tab_handle)
                                    driver.close()

                                    # Step 5: Switch back to the original window/tab
                                    driver.switch_to.window(original_window)

                    except Exception as err:
                        print("Error " + name_content, err)
                        file.write("Error " + name_content + "\n")
                        # driver.switch_to.window(driver.window_handles[-1])
                else:
                    print(newfilename + " already exists")

    # Close the CSV file
    csv_file.close()
    file.close()


retrieve_by_query(driver, "Comirnaty")
